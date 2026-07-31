"""Runner da suíte de avaliação do assistente de RH.

Carrega os casos de avaliacao/casos.json, invoca o grafo uma vez por caso e
confronta o estado final com o gabarito declarado no caso (ver avaliacao/regua.py).
Não há framework de teste envolvido: um caso é um dicionário de dados e a suíte
é um laço sobre eles.

O grafo é compilado SEM checkpointer. Quando um caso precisa de memória de
conversa, o histórico é declarado no próprio caso e entra pelo estado inicial —
o campo `mensagens` usa o reducer add_messages, que aceita mensagens já no
input e as trata como o histórico do turno.

Execução (dentro do container, onde as dependências estão instaladas):

    docker compose exec backend python -m avaliacao.rodar
    docker compose exec backend python -m avaliacao.rodar --caso tool-saldo-ana
    docker compose exec backend python -m avaliacao.rodar --caso tool-saldo-ana --repeticoes 5

Encerra com status 1 se algum caso falhar, para servir de porta em CI.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from app.graph import compilar_grafo
from avaliacao import regua

CASOS_PATH = Path(__file__).resolve().parent / "casos.json"

# Largura da coluna do nome do caso na saída, para o veredito ficar alinhado.
COLUNA = 62

# Mesmos campos de reset que a rota /api/chat monta a cada turno: sem zerá-los,
# um caso herdaria os intermediários do caso anterior. `mensagens` fica de fora
# de propósito — é o único campo que acumula, e é por ele que o histórico
# declarado no caso entra no estado.
RESET = {
    "categoria_triagem": "",
    "tipo_consulta": "",
    "consulta": "",
    "tentativas": 0,
    "ai_msg": None,
    "chunks": [],
    "tool_messages": [],
    "pergunta_dados": "",
    "pergunta_politica": "",
    "resposta_politica": None,
    "resposta_dados": None,
    "trajetoria": None,
    "groundedness_score": 0.0,
}

# Papéis aceitos no histórico declarado de um caso, mapeados para o tipo de
# mensagem correspondente do LangChain.
PAPEIS = {"usuario": HumanMessage, "assistente": AIMessage}


# --- Carga dos casos ---------------------------------------------------------

def carregar_casos(path: Path = CASOS_PATH) -> list[dict]:
    """Lê os casos do arquivo de dados.

    Os casos são dado, não código: acrescentar um caso é editar o JSON, sem
    tocar no runner.
    """
    dados = json.loads(path.read_text(encoding="utf-8"))
    return dados["casos"]


def montar_historico(caso: dict) -> list:
    """Converte o histórico declarado no caso em mensagens do LangChain."""
    return [PAPEIS[m["papel"]](content=m["texto"]) for m in caso.get("historico", [])]


def estado_inicial(caso: dict) -> dict:
    """Monta o estado inicial do grafo para um caso."""
    entradas = caso.get("entradas", {})
    return {
        **RESET,
        "pergunta": caso["pergunta"],
        "auto_corrigir": entradas.get("auto_corrigir", False),
        "validar_teto": entradas.get("validar_teto", False),
        "mensagens": montar_historico(caso),
    }


# --- Execução ----------------------------------------------------------------

def executar(grafo, caso: dict, repeticao: int = 0) -> dict:
    """Invoca o grafo para um caso e devolve o estado final completo."""
    return grafo.invoke(
        estado_inicial(caso),
        # thread_id por caso e repetição: sem checkpointer ele é ignorado, mas
        # mantém o config no mesmo formato usado em produção e garante que uma
        # repetição nunca herde estado de outra.
        config={"configurable": {"thread_id": f"avaliacao-{caso['id']}-{repeticao}"}},
    )


def avaliar_caso(grafo, caso: dict, repeticao: int = 0):
    """Executa um caso e aplica a régua. Devolve (vereditos, nao_avaliados, estado).

    Erro na invocação vira um veredito reprovado, para uma exceção não passar
    por caso aprovado nem derrubar a suíte inteira.
    """
    try:
        estado = executar(grafo, caso, repeticao)
    except Exception as exc:
        falha = regua.Veredito("execucao", False, f"{type(exc).__name__}: {exc}")
        return [falha], [], None
    vereditos, nao_avaliados = regua.avaliar(estado, caso.get("criterios", {}))
    return vereditos, nao_avaliados, estado


# --- Saída -------------------------------------------------------------------

def _cabecalho(nome: str, veredito: str) -> str:
    """Nome do caso com preenchimento pontilhado até a coluna do veredito."""
    return f"{nome} {'.' * max(3, COLUNA - len(nome))} {veredito}"


def imprimir_execucao_unica(caso, vereditos, nao_avaliados, estado) -> None:
    """Imprime o resultado de um caso executado uma vez."""
    passou = all(v.ok for v in vereditos)
    print()
    print(_cabecalho(caso["id"], "PASSOU" if passou else "FALHOU"))
    for v in vereditos:
        marca = "✓" if v.ok else "✗"
        print(f"  {marca} {v.criterio:<13}{v.detalhe}")
    for nome in nao_avaliados:
        print(f"  · {nome:<13}(nenhum critério registrado — não avaliado)")
    if estado is not None:
        # O groundedness é sempre reportado e nunca decide passou/falhou: é uma
        # similaridade cosseno, cega a negação e zerada fora da rota de política.
        print(f"  · {'groundedness':<13}{estado.get('groundedness_score', 0.0)}  (reportado, nunca decide)")


def imprimir_repeticoes(caso, rodadas: list[list[regua.Veredito]], nao_avaliados) -> None:
    """Imprime o resultado agregado de um caso executado N vezes."""
    total = len(rodadas)
    passaram = sum(1 for vereditos in rodadas if all(v.ok for v in vereditos))
    print()
    print(_cabecalho(caso["id"], f"{passaram}/{total} passou"))
    # Só os critérios que reprovaram em alguma rodada, com a contagem: é o que
    # identifica qual critério oscila.
    falhas: dict[str, list[str]] = {}
    for vereditos in rodadas:
        for v in vereditos:
            if not v.ok:
                falhas.setdefault(v.criterio, []).append(v.detalhe)
    if not falhas:
        print(f"  ✓ {'todos':<13}{total} de {total} rodadas com todos os critérios aprovados")
    for criterio, detalhes in falhas.items():
        print(f"  ✗ {criterio:<13}({len(detalhes)} de {total}) {detalhes[0]}")
    for nome in nao_avaliados:
        print(f"  · {nome:<13}(nenhum critério registrado — não avaliado)")


# --- Entrada -----------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Suíte de avaliação do assistente de RH.")
    p.add_argument("--caso", help="Executa apenas o caso com este id.")
    p.add_argument(
        "--repeticoes", type=int, default=1,
        help="Executa cada caso N vezes e reporta quantas passaram (expõe oscilação).",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    casos = carregar_casos()

    if args.caso:
        casos = [c for c in casos if c["id"] == args.caso]
        if not casos:
            print(f"Nenhum caso com id {args.caso!r}.")
            return 1

    grafo = compilar_grafo()

    sufixo = f" × {args.repeticoes} repetições" if args.repeticoes > 1 else ""
    print(f"Suíte de avaliação — {len(casos)} casos{sufixo}")

    reprovados = []
    for caso in casos:
        if args.repeticoes > 1:
            rodadas = []
            nao_avaliados: list[str] = []
            for i in range(args.repeticoes):
                vereditos, nao_avaliados, _ = avaliar_caso(grafo, caso, repeticao=i)
                rodadas.append(vereditos)
            imprimir_repeticoes(caso, rodadas, nao_avaliados)
            if not all(all(v.ok for v in r) for r in rodadas):
                reprovados.append(caso["id"])
        else:
            vereditos, nao_avaliados, estado = avaliar_caso(grafo, caso)
            imprimir_execucao_unica(caso, vereditos, nao_avaliados, estado)
            if not all(v.ok for v in vereditos):
                reprovados.append(caso["id"])

    aprovados = len(casos) - len(reprovados)
    print()
    print("─" * (COLUNA + 12))
    print(f"{aprovados}/{len(casos)} casos passaram")
    if reprovados:
        print(f"reprovados: {', '.join(reprovados)}")
    return 1 if reprovados else 0


if __name__ == "__main__":
    sys.exit(main())
