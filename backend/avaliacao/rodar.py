"""Runner da suíte de avaliação do assistente de RH.

Carrega os casos de avaliacao/casos.json, invoca o grafo uma vez por caso e
confronta o estado final com o gabarito declarado no caso: os critérios com
gabarito vão para a régua determinística (avaliacao/regua.py) e o que só se
julga lendo o texto da resposta vai para o juiz (avaliacao/juiz.py). Não há
framework de teste envolvido: um caso é um dicionário de dados e a suíte é um
laço sobre eles.

O grafo é compilado SEM checkpointer. Quando um caso precisa de memória de
conversa, o histórico é declarado no próprio caso e entra pelo estado inicial —
o campo `mensagens` usa o reducer add_messages, que aceita mensagens já no
input e as trata como o histórico do turno.

Execução (dentro do container, onde as dependências estão instaladas):

    docker compose exec backend python -m avaliacao.rodar
    docker compose exec backend python -m avaliacao.rodar --rapido
    docker compose exec backend python -m avaliacao.rodar --caso tool-saldo-ana
    docker compose exec backend python -m avaliacao.rodar --caso tool-saldo-ana --repeticoes 5

Ao fim de cada rodada o runner compara o resultado com o da rodada anterior e
imprime o delta, no formato "5/5 → 3/5, 2 regressões". A base de comparação fica
em .ultima-rodada.json, fora do git.

Encerra com status 1 se algum caso falhar, para servir de porta em CI.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from app.graph import compilar_grafo
from avaliacao import juiz, regua

# Critérios que alguém sabe medir: os determinísticos da régua e o do juiz. Um
# critério declarado num caso e ausente daqui aparece na saída como não
# avaliado, para não passar por aprovado no silêncio.
RECONHECIDOS = set(regua.CRITERIOS) | {juiz.CRITERIO}

CASOS_PATH = Path(__file__).resolve().parent / "casos.json"

# Resultado da última rodada, base de comparação do delta. Fica fora do git
# (ver .gitignore): é estado local de quem roda a suíte, não conteúdo do projeto.
ULTIMA_PATH = Path(__file__).resolve().parent / ".ultima-rodada.json"

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


def selecionar(casos: list[dict], caso_id: str | None, rapido: bool) -> list[dict]:
    """Aplica os filtros de seleção, que compõem entre si."""
    if rapido:
        casos = [c for c in casos if c.get("rapido")]
    if caso_id:
        casos = [c for c in casos if c["id"] == caso_id]
    return casos


# --- Comparação com a rodada anterior ----------------------------------------

def carregar_ultima() -> dict[str, bool]:
    """Lê o resultado por caso da rodada anterior. Ausente ou ilegível → vazio."""
    try:
        return json.loads(ULTIMA_PATH.read_text(encoding="utf-8"))["resultados"]
    except Exception:
        return {}


def gravar_ultima(anterior: dict[str, bool], atual: dict[str, bool]) -> None:
    """Grava o resultado por caso, mesclando com o que já havia.

    A mescla preserva o resultado de casos que não rodaram nesta seleção — sem
    ela, uma rodada no modo rápido apagaria a base de comparação dos demais.
    """
    ULTIMA_PATH.write_text(
        json.dumps({"resultados": {**anterior, **atual}}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def imprimir_delta(anterior: dict[str, bool], atual: dict[str, bool]) -> None:
    """Imprime o delta contra a rodada anterior.

    A comparação cobre só os casos com resultado registrado antes E agora:
    comparar contagens de conjuntos diferentes de casos daria um número que
    parece medir regressão e não mede.
    """
    comparaveis = [cid for cid in atual if cid in anterior]
    sem_base = len(atual) - len(comparaveis)
    if not comparaveis:
        print("delta: sem rodada anterior para comparar")
        return
    antes = sum(1 for cid in comparaveis if anterior[cid])
    agora = sum(1 for cid in comparaveis if atual[cid])
    regressoes = [cid for cid in comparaveis if anterior[cid] and not atual[cid]]
    recuperacoes = [cid for cid in comparaveis if not anterior[cid] and atual[cid]]
    total = len(comparaveis)
    print(f"delta: {antes}/{total} → {agora}/{total}, {len(regressoes)} regressões")
    if regressoes:
        print(f"  regrediram:  {', '.join(regressoes)}")
    if recuperacoes:
        print(f"  recuperaram: {', '.join(recuperacoes)}")
    if sem_base:
        print(f"  ({sem_base} caso(s) sem rodada anterior, fora da comparação)")


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
    """Executa um caso e aplica os critérios. Devolve (vereditos, nao_avaliados, estado).

    Erro na invocação vira um veredito reprovado, para uma exceção não passar
    por caso aprovado nem derrubar a suíte inteira.
    """
    criterios = caso.get("criterios", {})
    try:
        estado = executar(grafo, caso, repeticao)
    except Exception as exc:
        falha = regua.Veredito("execucao", False, f"{type(exc).__name__}: {exc}")
        return [falha], [], None
    vereditos = regua.avaliar(estado, criterios)
    # O juiz roda depois da régua, mas não recebe o resultado dela: saber que os
    # critérios determinísticos passaram o inclinaria a concordar com eles.
    if juiz.CRITERIO in criterios:
        vereditos.append(juiz.avaliar(caso, estado, criterios[juiz.CRITERIO]))
    nao_avaliados = [nome for nome in criterios if nome not in RECONHECIDOS]
    return vereditos, nao_avaliados, estado


# --- Saída -------------------------------------------------------------------

def _cabecalho(nome: str, veredito: str) -> str:
    """Nome do caso com preenchimento pontilhado até a coluna do veredito."""
    return f"{nome} {'.' * max(3, COLUNA - len(nome))} {veredito}"


def passou_caso(vereditos: list[regua.Veredito]) -> bool:
    """Um caso passa quando todo critério EFETIVAMENTE MEDIDO foi aprovado.

    Critério não avaliado (juiz indisponível) não reprova o caso: a suíte não
    acusa defeito no sistema por causa de um problema do avaliador.
    """
    return all(v.ok for v in vereditos if v.avaliado)


def imprimir_execucao_unica(caso, vereditos, nao_avaliados, estado) -> None:
    """Imprime o resultado de um caso executado uma vez."""
    print()
    print(_cabecalho(caso["id"], "PASSOU" if passou_caso(vereditos) else "FALHOU"))
    for v in vereditos:
        marca = "✓" if v.ok else "✗"
        if not v.avaliado:
            marca = "·"
        print(f"  {marca} {v.criterio:<13}{v.detalhe}")
    for nome in nao_avaliados:
        print(f"  · {nome:<13}(nenhum critério registrado — não avaliado)")
    if estado is not None:
        # O groundedness é sempre reportado e nunca decide passou/falhou: é uma
        # similaridade cosseno, cega a negação e zerada fora da rota de política.
        print(f"  · {'groundedness':<13}{estado.get('groundedness_score', 0.0)}  (reportado, nunca decide)")


def imprimir_repeticoes(caso, rodadas: list[list[regua.Veredito]], nao_avaliados) -> None:
    """Imprime o resultado agregado de um caso executado N vezes.

    Relata todo critério que reprovou em alguma rodada, distinguindo o que
    reprova sempre do que OSCILOU entre rodadas — um critério que muda de
    veredito com a mesma entrada é informação, não ruído a ser suprimido.
    """
    total = len(rodadas)
    passaram = sum(1 for vereditos in rodadas if passou_caso(vereditos))
    print()
    print(_cabecalho(caso["id"], f"{passaram}/{total} passou"))

    por_criterio: dict[str, list[regua.Veredito]] = {}
    for vereditos in rodadas:
        for v in vereditos:
            por_criterio.setdefault(v.criterio, []).append(v)

    if passaram == total:
        print(f"  ✓ {'todos':<13}{total} de {total} rodadas com todos os critérios aprovados")

    for criterio, vs in por_criterio.items():
        nao_medidos = [v for v in vs if not v.avaliado]
        if nao_medidos:
            print(f"  · {criterio:<13}não avaliado em {len(nao_medidos)} de {total} rodadas")
            for detalhe in sorted({v.detalhe for v in nao_medidos}):
                print(f"      {detalhe}")
        medidos = [v for v in vs if v.avaliado]
        resultados = {v.ok for v in medidos}
        if not medidos or resultados == {True}:
            continue  # nada medido, ou estável e aprovado: nada a relatar
        reprovas = sum(1 for v in medidos if not v.ok)
        oscilou = len(resultados) > 1
        marca = "!" if oscilou else "✗"
        rotulo = (
            f"OSCILOU — {reprovas} de {len(medidos)} rodadas medidas reprovaram"
            if oscilou
            else f"({reprovas} de {len(medidos)})"
        )
        print(f"  {marca} {criterio:<13}{rotulo}")
        # Justificativas distintas: num critério que oscilou, é a comparação
        # entre elas que mostra o que mudou de uma rodada para a outra.
        vistos = {f"[{'aprovado' if v.ok else 'reprovado'}] {v.detalhe}" for v in medidos}
        for detalhe in sorted(vistos):
            print(f"      {detalhe}")

    for nome in nao_avaliados:
        print(f"  · {nome:<13}(nenhum critério registrado — não avaliado)")


# --- Entrada -----------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Suíte de avaliação do assistente de RH.")
    p.add_argument("--caso", help="Executa apenas o caso com este id.")
    p.add_argument(
        "--rapido", action="store_true",
        help="Executa só o conjunto de demonstração (casos marcados com \"rapido\" em casos.json).",
    )
    p.add_argument(
        "--repeticoes", type=int, default=1,
        help="Executa cada caso N vezes e reporta quantas passaram (expõe oscilação).",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    casos = selecionar(carregar_casos(), args.caso, args.rapido)
    if not casos:
        print("Nenhum caso corresponde à seleção.")
        return 1

    grafo = compilar_grafo()

    escopo = "conjunto rápido" if args.rapido else f"{len(casos)} casos"
    sufixo = f" × {args.repeticoes} repetições" if args.repeticoes > 1 else ""
    print(f"Suíte de avaliação — {escopo}{sufixo}")

    resultados: dict[str, bool] = {}
    for caso in casos:
        if args.repeticoes > 1:
            rodadas = []
            nao_avaliados: list[str] = []
            for i in range(args.repeticoes):
                vereditos, nao_avaliados, _ = avaliar_caso(grafo, caso, repeticao=i)
                rodadas.append(vereditos)
            imprimir_repeticoes(caso, rodadas, nao_avaliados)
            resultados[caso["id"]] = all(passou_caso(r) for r in rodadas)
        else:
            vereditos, nao_avaliados, estado = avaliar_caso(grafo, caso)
            imprimir_execucao_unica(caso, vereditos, nao_avaliados, estado)
            resultados[caso["id"]] = passou_caso(vereditos)

    reprovados = [cid for cid, ok in resultados.items() if not ok]
    aprovados = len(casos) - len(reprovados)
    print()
    print("─" * (COLUNA + 12))
    print(f"{aprovados}/{len(casos)} casos passaram")
    if reprovados:
        print(f"reprovados: {', '.join(reprovados)}")

    # Regressão se mede contra a rodada anterior, não contra um número absoluto.
    # Só o modo de execução única alimenta a base: uma rodada com repetições usa
    # um critério mais estrito (todas as rodadas têm de passar) e misturar os
    # dois na mesma base tornaria o delta incomparável.
    anterior = carregar_ultima()
    imprimir_delta(anterior, resultados)
    if args.repeticoes == 1:
        gravar_ultima(anterior, resultados)

    return 1 if reprovados else 0


if __name__ == "__main__":
    sys.exit(main())
