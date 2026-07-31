"""Runner da suíte de avaliação do assistente de RH.

Carrega os casos de avaliacao/casos.json, invoca o grafo uma vez por caso e
imprime o que cada execução produziu. Não há framework de teste envolvido: um
caso é um dicionário de dados e a suíte é um laço sobre eles.

O grafo é compilado SEM checkpointer. Quando um caso precisa de memória de
conversa, o histórico é declarado no próprio caso e entra pelo estado inicial —
o campo `mensagens` usa o reducer add_messages, que aceita mensagens já no
input e as trata como o histórico do turno.

Execução (dentro do container, onde as dependências estão instaladas):

    docker compose exec backend python -m avaliacao.rodar

Encerra com status 1 se algum caso falhar, para servir de porta em CI.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from app.graph import compilar_grafo

CASOS_PATH = Path(__file__).resolve().parent / "casos.json"

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

def executar(grafo, caso: dict) -> dict:
    """Invoca o grafo para um caso e devolve o estado final completo."""
    return grafo.invoke(
        estado_inicial(caso),
        # thread_id por caso: sem checkpointer ele é ignorado, mas mantém o
        # config no mesmo formato usado em produção.
        config={"configurable": {"thread_id": f"avaliacao-{caso['id']}"}},
    )


# --- Saída -------------------------------------------------------------------

def formatar_rota(trajetoria: list[str]) -> str:
    """Trajetória do turno como uma sequência legível de nós."""
    return " → ".join(trajetoria)


def imprimir_caso(caso: dict, estado: dict) -> None:
    """Imprime o que a execução de um caso produziu."""
    resposta = estado["resposta"]
    print(f"\n{caso['id']}")
    print(f"  pergunta      {caso['pergunta']}")
    print(f"  rota          {formatar_rota(estado.get('trajetoria', []))}")
    print(f"  fontes        {', '.join(f.arquivo for f in resposta.fontes) or '(nenhuma)'}")
    print(f"  categoria     {resposta.categoria}")
    # O groundedness é sempre REPORTADO e nunca decide passou/falhou: é uma
    # similaridade cosseno, cega a negação e zerada fora da rota de política.
    print(f"  groundedness  {estado.get('groundedness_score', 0.0)}  (reportado, não é critério)")
    print(f"  resposta      {resposta.resposta.strip().splitlines()[0][:100]}")


def imprimir_erro(caso: dict, exc: Exception) -> None:
    print(f"\n{caso['id']}")
    print(f"  pergunta      {caso['pergunta']}")
    print(f"  ERRO          {type(exc).__name__}: {exc}")


# --- Entrada -----------------------------------------------------------------

def main() -> int:
    casos = carregar_casos()
    grafo = compilar_grafo()

    print(f"Suíte de avaliação — {len(casos)} casos")

    falhas = 0
    for caso in casos:
        try:
            estado = executar(grafo, caso)
        except Exception as exc:
            imprimir_erro(caso, exc)
            falhas += 1
            continue
        imprimir_caso(caso, estado)

    executados = len(casos) - falhas
    print(f"\n{'─' * 60}")
    print(f"{executados}/{len(casos)} casos executados sem erro")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
