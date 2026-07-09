"""
Fluxo de decisão do assistente de RH modelado como grafo (LangGraph).

O atendimento continua tendo dois caminhos que não cabem numa única chamada ao
modelo (tool calling e structured output se excluem — ver nota abaixo), mas
agora cada etapa é um nó explícito de um StateGraph, com o estado fluindo entre
eles em vez de variáveis locais:

    START → decidir_rota → (tem tool_calls?)
        ├── não → recuperar → gerar → validar_fontes → END   (rota informativa)
        └── sim → executar_tools → formatar_tool → END        (rota de tool)

1. decidir_rota (bind_tools): o modelo recebe a pergunta com as tools plugadas
   e decide. Sem tool → pergunta de política. Com tool(s) → executa.
2a. Rota informativa: recupera os chunks relevantes por similaridade (retrieval)
    e formula a resposta só com base neles.
2b. Rota de tool: executa cada tool pedida e devolve o resultado ao modelo, que
    formata a resposta final.

Nota sobre tool calling + structured output: with_structured_output já é
implementado forçando uma tool call para o schema RespostaRH, o que conflita com
bind_tools na mesma chamada — o modelo seria forçado a "responder" e não poderia
decidir por uma tool de negócio. Por isso a decisão/execução e a formatação em
RespostaRH ficam em nós separados.

Requer ANTHROPIC_API_KEY (geração) e OPENAI_API_KEY (embeddings da busca) no
ambiente (ver .env / docker-compose.yml).
"""
from __future__ import annotations

import logging
from typing import Any, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from app.retrieval import TOP_K, buscar
from app.schemas import Fonte, RespostaRH
from app.tools import (
    alerta_teto,
    consultar_saldo_ferias,
    registrar_solicitacao_ferias,
)

logger = logging.getLogger(__name__)


# --- Estado do grafo --------------------------------------------------------

class EstadoRH(TypedDict, total=False):
    """Estado que flui entre os nós do grafo.

    TypedDict simples (sem Annotated/reducers): cada nó devolve um dict parcial
    e o LangGraph sobrescreve as chaves atualizadas. Os campos cobrem tanto a
    entrada (pergunta e as flags de UI) quanto os valores intermediários que
    antes eram variáveis locais do fluxo.
    """

    # Entrada
    pergunta: str
    reescrever_pergunta: bool
    validar_teto: bool
    # Intermediários
    consulta: str  # pergunta usada na busca/geração (reescrita, se a flag ligar)
    ai_msg: Any  # AIMessage da etapa de decisão (carrega os tool_calls)
    chunks: list[Document]  # chunks recuperados na rota informativa
    tool_messages: list[ToolMessage]  # resultados das tools na rota de tool
    # Saída
    resposta: RespostaRH


# --- Prompts ----------------------------------------------------------------

SYSTEM = (
    "Você é um assistente de RH. Responda de forma clara e objetiva, SOMENTE "
    "com base no contexto fornecido (trechos recuperados das políticas), sem "
    "inventar nem usar conhecimento externo.\n\n"
    "Interprete a pergunta pelo significado, não pela coincidência de palavras: "
    "se o contexto trata do assunto perguntado, use-o para responder MESMO que "
    "a pergunta empregue sinônimos ou perífrases (ex.: \"descanso remunerado\", "
    "\"período de descanso\" e \"afastamento remunerado\" referem-se a férias). "
    "A ausência das mesmas palavras no texto NÃO é motivo para recusar.\n"
    "Responda exatamente \"Não encontrei essa informação na base.\" e sugira "
    "procurar o RH apenas quando o contexto recuperado for realmente irrelevante "
    "para a pergunta (não cobre o assunto). Nesse caso, deixe a lista de fontes "
    "vazia. Sempre cite de qual política veio a informação.\n\n"
    "Preencha a resposta estruturada assim:\n"
    "- fontes: apenas as políticas que você de fato usou para responder "
    "(arquivo e titulo exatamente como aparecem no contexto). Se não usou "
    "nenhuma, deixe a lista vazia.\n"
    "- categoria: a categoria da pergunta (ferias, home-office, beneficios, "
    "reembolso, horario, licencas ou outro).\n"
    "- confianca: de 0 a 1, o quão confiante você está na resposta com base "
    "no contexto."
)

# Etapa de decisão: system que orienta o modelo a decidir entre tool e política.
SYSTEM_DECISAO = (
    "Você é um assistente de RH com duas ferramentas:\n"
    "- consultar_saldo_ferias: para saber quantos dias um funcionário tem ou "
    "usou (ex.: \"quantos dias a Ana tem?\").\n"
    "- registrar_solicitacao_ferias: para AGENDAR/SOLICITAR férias.\n\n"
    "Regra de decisão:\n"
    "1. Se a pergunta tiver INTENÇÃO DE AGENDAR/SOLICITAR férias, você DEVE "
    "chamar registrar_solicitacao_ferias — NÃO explique a política. São sinais "
    "de intenção: verbos como \"solicitar\", \"registrar\", \"agendar\", "
    "\"marcar\", \"quero tirar\", \"gostaria de tirar\", combinados com um nome "
    "de funcionário e/ou uma quantidade de dias e/ou um período (mês/datas).\n"
    "2. Se a pergunta for sobre o SALDO de um funcionário nominal, chame "
    "consultar_saldo_ferias.\n"
    "3. Se a pergunta for INFORMATIVA sobre as regras (ex.: \"como funciona\", "
    "\"quantos dias tenho direito\", \"qual a regra\"), responda normalmente, "
    "SEM ferramenta — outra etapa formula a resposta pela política.\n\n"
    "Exemplos:\n"
    "- \"Quero solicitar 5 dias de férias em julho para o Bruno\" → chama "
    "registrar_solicitacao_ferias (é uma solicitação de agendamento).\n"
    "- \"Quantos dias de férias eu tenho direito?\" → não chama ferramenta "
    "(é informativa sobre a regra)."
)

# Rota de tool: system que orienta a formatar a resposta final a partir do
# resultado da(s) ferramenta(s).
SYSTEM_FORMATA_TOOL = (
    "Você é um assistente de RH. Use o resultado das ferramentas para formular "
    "a resposta final ao usuário, de forma clara e objetiva.\n\n"
    "Confirme apenas o que de fato aconteceu, com base no resultado da "
    "ferramenta. Não prometa etapas futuras que o sistema não realiza: ao "
    "registrar uma solicitação de férias, apenas confirme que ela foi "
    "registrada (funcionário, dias e período). Não diga que o saldo será "
    "verificado nem que a solicitação será processada ou aprovada.\n\n"
    "Preencha a resposta estruturada assim:\n"
    "- fontes: deixe a lista vazia (a informação veio de um sistema interno, "
    "não de um documento de política).\n"
    "- categoria: a categoria da pergunta (use 'ferias' para saldo ou "
    "solicitação de férias).\n"
    "- confianca: de 0 a 1, sua confiança na resposta."
)

# Reescrita de consulta: normaliza a pergunta removendo ruído (artigos/palavras
# supérfluas), que desloca o vetor da pergunta e pode reordenar o top-k (um termo
# a mais aproxima um chunk irrelevante e afasta o certo). A query normalizada
# alimenta a busca e a geração, preservando o sentido original.
SYSTEM_REESCRITA = (
    "Você reescreve a pergunta do usuário para uma consulta de busca semântica. "
    "Deixe-a clara e neutra, removendo artigos e palavras desnecessárias, mas "
    "preservando integralmente o sentido. Responda SOMENTE com a consulta "
    "reescrita, sem explicação, aspas ou texto adicional."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        ("human", "Contexto:\n{contexto}\n\nPergunta: {pergunta}"),
    ]
)


# --- Modelo, tools e chains -------------------------------------------------

model = ChatAnthropic(model="claude-haiku-4-5", temperature=0)

# Decisão: modelo com as tools plugadas (decide se chama alguma).
TOOLS = [consultar_saldo_ferias, registrar_solicitacao_ferias]
TOOLS_POR_NOME = {t.name: t for t in TOOLS}
model_com_tools = model.bind_tools(TOOLS)

# Formatação: modelo que sempre devolve RespostaRH.
model_estruturado = model.with_structured_output(RespostaRH)

# Chain de política (sem tool): prompt com contexto | saída estruturada.
chain = prompt | model_estruturado


# --- Helpers dos nós --------------------------------------------------------

def _formatar_contexto(chunks: list[Document]) -> str:
    """Monta o contexto a partir dos chunks recuperados.

    Cada bloco traz título e arquivo da política de origem, para o modelo poder
    se ancorar na fonte de cada informação. Só os chunks recuperados entram —
    não mais todos os documentos.
    """
    return "\n\n---\n\n".join(
        f"[Política: {c.metadata.get('titulo')} | arquivo: {c.metadata.get('arquivo')}]\n"
        f"{c.page_content}"
        for c in chunks
    )


def _reescrever_pergunta(pergunta: str) -> str:
    """Normaliza a pergunta para a busca (artigos/palavras supérfluas removidos).

    Usa o mesmo modelo da geração (temperature=0 → determinístico). Em caso de
    saída vazia ou falha, cai para a pergunta original — a reescrita é um reforço
    da busca, não pode degradar o caminho atual.
    """
    try:
        msg = model.invoke(
            [
                SystemMessage(content=SYSTEM_REESCRITA),
                HumanMessage(content=pergunta),
            ]
        )
        reescrita = (msg.content or "").strip()
        return reescrita or pergunta
    except Exception:
        return pergunta


def _validar_fontes(fontes: list[Fonte], chunks: list[Document]) -> list[Fonte]:
    """Filtra as fontes citadas pelo modelo contra os chunks recuperados.

    A seleção de fonte é do modelo — ele é quem "lê" os chunks e sabe quais de
    fato usou para responder (o top-k traz ruído, então citar todos seria
    impreciso). O código só faz a salvaguarda anti-alucinação: descarta qualquer
    arquivo citado que NÃO esteja entre os recuperados e canoniza o título a
    partir dos metadados (evita divergência de texto). Dedupe por arquivo.
    """
    titulos = {
        c.metadata.get("arquivo"): c.metadata.get("titulo")
        for c in chunks
        if c.metadata.get("arquivo")
    }
    validas: dict[str, Fonte] = {}
    for f in fontes:
        if f.arquivo in titulos and f.arquivo not in validas:
            validas[f.arquivo] = Fonte(arquivo=f.arquivo, titulo=titulos[f.arquivo] or f.titulo)
    return list(validas.values())


# --- Nós --------------------------------------------------------------------

def decidir_rota(state: EstadoRH) -> EstadoRH:
    """Etapa de decisão: o modelo (com tools plugadas) decide se usa alguma.

    O AIMessage resultante segue no estado: seus tool_calls definem a rota e,
    na rota de tool, ele é reinjetado na formatação final.
    """
    ai_msg = model_com_tools.invoke(
        [
            SystemMessage(content=SYSTEM_DECISAO),
            HumanMessage(content=state["pergunta"]),
        ]
    )
    return {"ai_msg": ai_msg}


def rota_apos_decisao(state: EstadoRH) -> str:
    """Aresta condicional: encaminha para a rota de tool ou a informativa.

    Sem tool_calls → pergunta de política (rota informativa). Com tool(s) → o
    código executa cada uma (rota de tool).
    """
    return "tool" if state["ai_msg"].tool_calls else "informativo"


def recuperar(state: EstadoRH) -> EstadoRH:
    """Define a consulta e recupera os chunks relevantes por similaridade.

    A reescrita (quando ligada) normaliza a pergunta, removendo ruído
    (artigos/palavras supérfluas). A query normalizada alimenta TANTO a busca
    (estabiliza o ranking do top-k) QUANTO a geração (uma pergunta limpa reduz
    recusas por ruído na própria pergunta).
    """
    consulta = state["pergunta"]
    if state.get("reescrever_pergunta"):
        consulta = _reescrever_pergunta(state["pergunta"])
        logger.info("[rewrite] %r -> %r", state["pergunta"], consulta)
    chunks = buscar(consulta, k=TOP_K)
    return {"consulta": consulta, "chunks": chunks}


def gerar(state: EstadoRH) -> EstadoRH:
    """Formula a resposta de política só com base nos chunks recuperados."""
    resposta = chain.invoke(
        {"contexto": _formatar_contexto(state["chunks"]), "pergunta": state["consulta"]}
    )
    return {"resposta": resposta}


def validar_fontes(state: EstadoRH) -> EstadoRH:
    """Salvaguarda anti-alucinação sobre as fontes citadas pelo modelo.

    As fontes citadas vêm do modelo (que sabe quais chunks usou); aqui só
    validamos contra o conjunto recuperado. Groundedness preservado: se não
    encontrou na base, o modelo deixa fontes vazias e a validação mantém vazio.
    """
    resposta = state["resposta"]
    chunks = state["chunks"]
    resposta.fontes = _validar_fontes(resposta.fontes, chunks)
    # Log de inspeção do retrieval: o conjunto recuperado traz os k chunks (com
    # ruído, de propósito), mas as fontes citadas devem ser um subconjunto fiel.
    # Registrar ambos permite observar essa diferença.
    recuperados = sorted(
        {c.metadata.get("arquivo") for c in chunks if c.metadata.get("arquivo")}
    )
    citados = [f.arquivo for f in resposta.fontes]
    logger.info(
        "retrieval informativo: pergunta=%r recuperados=%s citados=%s",
        state["pergunta"], recuperados, citados,
    )
    return {"resposta": resposta}


def executar_tools(state: EstadoRH) -> EstadoRH:
    """Executa cada tool pedida pelo modelo e devolve os resultados via ToolMessage."""
    ai_msg = state["ai_msg"]
    tool_messages: list[ToolMessage] = []
    for call in ai_msg.tool_calls:
        tool = TOOLS_POR_NOME.get(call["name"])
        resultado = (
            tool.invoke(call["args"]) if tool else "Ferramenta desconhecida"
        )
        # Validação opcional do teto da política (controlada pela UI): o sistema
        # sinaliza saldo acima do máximo, sem depender do modelo.
        if state.get("validar_teto") and call["name"] == "consultar_saldo_ferias":
            alerta = alerta_teto(call["args"].get("funcionario", ""))
            if alerta:
                resultado = f"{resultado} {alerta}"
        tool_messages.append(
            ToolMessage(content=str(resultado), tool_call_id=call["id"])
        )
    return {"tool_messages": tool_messages}


def formatar_tool(state: EstadoRH) -> EstadoRH:
    """Formula a resposta final como RespostaRH a partir do resultado das tools."""
    resposta = model_estruturado.invoke(
        [
            SystemMessage(content=SYSTEM_FORMATA_TOOL),
            HumanMessage(content=state["pergunta"]),
            state["ai_msg"],
            *state["tool_messages"],
        ]
    )
    return {"resposta": resposta}


# --- Montagem do grafo ------------------------------------------------------

def _construir_grafo():
    """Monta e compila o StateGraph do fluxo de decisão (API clássica)."""
    g = StateGraph(EstadoRH)
    g.add_node("decidir_rota", decidir_rota)
    g.add_node("recuperar", recuperar)
    g.add_node("gerar", gerar)
    g.add_node("validar_fontes", validar_fontes)
    g.add_node("executar_tools", executar_tools)
    g.add_node("formatar_tool", formatar_tool)

    g.add_edge(START, "decidir_rota")
    g.add_conditional_edges(
        "decidir_rota",
        rota_apos_decisao,
        {"informativo": "recuperar", "tool": "executar_tools"},
    )
    # Rota informativa (política).
    g.add_edge("recuperar", "gerar")
    g.add_edge("gerar", "validar_fontes")
    g.add_edge("validar_fontes", END)
    # Rota de tool.
    g.add_edge("executar_tools", "formatar_tool")
    g.add_edge("formatar_tool", END)

    return g.compile()


# Grafo compilado, pronto para invoke() (importado por app/chat.py).
grafo = _construir_grafo()
