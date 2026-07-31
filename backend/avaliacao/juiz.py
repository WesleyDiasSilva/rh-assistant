"""Critério julgado por modelo, para o que não tem gabarito.

A régua determinística cobre tudo que o sistema registra em campo estruturado:
rota, fontes, ferramenta escolhida, contadores. O que ela não alcança é o texto
da resposta — se ele derivou um número que não estava no histórico, se enunciou
uma regra que o contexto não traz. Aí entra o juiz.

Duas escolhas que definem o formato:

- O veredito é BINÁRIO ("aprovado" / "reprovado") acompanhado de justificativa,
  nunca uma nota de 1 a 5. Nota de escala pede um limiar arbitrário e convida a
  discutir se 3 passa, deslocando a conversa do defeito para o número. Binário
  obriga a decidir, e o motivo obriga a explicar.
- O critério de cada caso é uma pergunta fechada que descreve um DEFEITO. Se o
  defeito está presente, reprova; se não está, aprova.

O juiz recebe a pergunta, o contexto que estava disponível e a resposta. Não
recebe o veredito da régua: se soubesse que os critérios determinísticos
passaram, tenderia a concordar com eles.
"""
from __future__ import annotations

from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from avaliacao.regua import Veredito

# Nome do critério, como aparece em casos.json.
CRITERIO = "juiz"

# O modelo do juiz é declarado aqui, e não importado do grafo, mesmo sendo hoje
# o mesmo haiku com temperature=0: o juiz avalia o sistema, então não deve
# acompanhar silenciosamente uma troca de modelo no sistema avaliado.
MODELO_JUIZ = "claude-haiku-4-5"


class _VereditoJuiz(BaseModel):
    """Saída estruturada do juiz: decisão binária mais a justificativa."""

    veredito: Literal["aprovado", "reprovado"] = Field(
        description=(
            "'reprovado' se o defeito descrito no critério está presente na "
            "resposta; 'aprovado' se não está."
        )
    )
    motivo: str = Field(
        description=(
            "Uma frase dizendo o que na resposta sustenta o veredito, citando o "
            "trecho relevante."
        )
    )


SYSTEM_JUIZ = (
    "Você é o juiz de uma suíte de avaliação de um assistente de RH. Recebe uma "
    "pergunta, o contexto que estava disponível para respondê-la e a resposta "
    "que o assistente produziu.\n\n"
    "O critério é uma pergunta fechada que descreve um DEFEITO a procurar na "
    "resposta:\n"
    "- Se o defeito ESTÁ presente, o veredito é \"reprovado\".\n"
    "- Se o defeito NÃO está presente, o veredito é \"aprovado\".\n\n"
    "Julgue somente o critério enunciado. Não avalie estilo, tom, extensão, "
    "cordialidade nem qualquer outra coisa que o critério não pergunte — uma "
    "resposta pode ser seca ou incompleta e ainda assim estar aprovada no "
    "critério em questão.\n\n"
    "Não avalie COMPLETUDE e não sugira o que a resposta deveria ter dito. Não "
    "é defeito a resposta deixar de mencionar algo, ser mais curta do que você "
    "escreveria ou não oferecer alternativas — a menos que o critério pergunte "
    "exatamente isso. Você julga o que a resposta AFIRMA, não o que ela omite.\n\n"
    "Baseie-se apenas na pergunta, no contexto e na resposta fornecidos. Não "
    "use conhecimento externo e não suponha contexto que não foi mostrado.\n\n"
    "No campo motivo, diga em uma frase o que na resposta sustenta o veredito, "
    "citando o trecho relevante. Não repita o enunciado do critério."
)

_modelo = ChatAnthropic(model=MODELO_JUIZ, temperature=0).with_structured_output(
    _VereditoJuiz
)


def montar_contexto(caso: dict, estado: dict) -> str:
    """Monta o contexto que estava disponível ao assistente, conforme o caso.

    Na rota de política o contexto são os chunks recuperados; nas rotas que
    respondem a partir da conversa, é o histórico declarado no caso. Quando não
    há nenhum dos dois, isso é dito explicitamente — o juiz precisa saber que a
    resposta não tinha material em que se apoiar.
    """
    chunks = estado.get("chunks") or []
    if chunks:
        return "\n\n---\n\n".join(
            f"[{c.metadata.get('arquivo')}]\n{c.page_content}" for c in chunks
        )
    historico = caso.get("historico") or []
    if historico:
        return "\n".join(f"{m['papel']}: {m['texto']}" for m in historico)
    return "(nenhum contexto: a resposta não se apoiou em documento nem em histórico)"


def avaliar(caso: dict, estado: dict, criterio: str) -> Veredito:
    """Submete a resposta ao juiz e devolve o veredito no formato da régua.

    Falha na chamada (rede, cota, indisponibilidade) marca o critério como não
    avaliado, e não como reprovado: a suíte não pode acusar defeito no sistema
    por causa de um problema do avaliador. O critério aparece na saída com o
    motivo da falha, sem entrar na conta de passou/falhou.
    """
    contexto = montar_contexto(caso, estado)
    resposta = estado["resposta"].resposta
    conteudo = (
        f"Critério: {criterio}\n\n"
        f"Pergunta: {caso['pergunta']}\n\n"
        f"Contexto disponível:\n{contexto}\n\n"
        f"Resposta do assistente:\n{resposta}"
    )
    try:
        julgado = _modelo.invoke(
            [SystemMessage(content=SYSTEM_JUIZ), HumanMessage(content=conteudo)]
        )
    except Exception as exc:
        return Veredito(
            CRITERIO,
            False,
            f"juiz indisponível ({type(exc).__name__}: {exc})",
            avaliado=False,
        )
    return Veredito(CRITERIO, julgado.veredito == "aprovado", julgado.motivo.strip())
