"""
Fluxo de decisão do assistente de RH modelado como grafo (LangGraph).

O atendimento continua tendo dois caminhos que não cabem numa única chamada ao
modelo (tool calling e structured output se excluem — ver nota abaixo), mas
agora cada etapa é um nó explícito de um StateGraph, com o estado fluindo entre
eles em vez de variáveis locais. Uma triagem na entrada desvia o que está fora
do escopo antes de gastar retrieval/tools:

    START → triagem → (é assunto de RH?)
        ├── não → resposta_direta → END                       (fora de escopo)
        └── sim → decidir_rota → (tem tool_calls?)
                ├── não → recuperar → gerar → validar_fontes → (sustentou?)
                │         ├── não, e auto_corrigir e < 1 retry → reescrever → recuperar
                │         └── sim, ou sem retry disponível → END       (informativa)
                └── sim → executar_tools → formatar_tool → END         (rota de tool)

0. triagem: classifica a pergunta em "rh" ou "fora_de_escopo" (saudações/small
   talk = fora_de_escopo). Fora de escopo vai para resposta_direta, que responde
   com educação dentro do papel do assistente, sem retrieval nem tools.
1. decidir_rota (bind_tools): o modelo recebe a pergunta com as tools plugadas
   e decide. Sem tool → pergunta de política. Com tool(s) → executa.
2a. Rota informativa: recupera os chunks relevantes por similaridade (retrieval)
    e formula a resposta só com base neles. Se a resposta não se sustentou na
    base (nenhuma fonte) e a auto-correção está ligada, o nó reescrever normaliza
    a pergunta e o ciclo tenta uma única vez mais (teto de tentativas garante
    terminação) — a reescrita é reação a falha, não pré-processamento.
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
from typing import Any, Literal, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

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
    auto_corrigir: bool
    validar_teto: bool
    # Intermediários
    categoria_triagem: str  # "rh" ou "fora_de_escopo" (definido pelo nó triagem)
    consulta: str  # pergunta usada na busca/geração (reescrita, após auto-correção)
    tentativas: int  # nº de reescritas já feitas (teto para a auto-correção terminar)
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

# Triagem de entrada: classifica se a pergunta é assunto de RH ou está fora do
# escopo (saudações/small talk incluídos). Serve para desviar o que não é RH
# antes de gastar retrieval/tools, respondendo direto e com educação.
SYSTEM_TRIAGEM = (
    "Você é a triagem de um assistente de RH interno. Classifique a mensagem do "
    "usuário em uma de duas categorias:\n"
    "- \"rh\": qualquer assunto de recursos humanos da empresa — políticas "
    "(férias, home-office, benefícios, reembolso, horário, licenças), saldo de "
    "férias de um funcionário ou solicitação/agendamento de férias.\n"
    "- \"fora_de_escopo\": saudações e conversa fiada (small talk) e temas "
    "claramente alheios ao trabalho (clima, piadas, esportes, notícias).\n\n"
    "Qualquer pergunta sobre temas de trabalho, empresa, escritório ou condições "
    "de trabalho é \"rh\", MESMO que o assunto pareça incomum. Na dúvida entre as "
    "duas, classifique como \"rh\" — o fluxo normal sabe recusar o que não está "
    "na base.\n\n"
    "Exemplos:\n"
    "- \"quero solicitar férias\" → rh\n"
    "- \"quantos dias a Ana tem?\" → rh\n"
    "- \"posso trazer meu cachorro pro escritório?\" → rh\n"
    "- \"qual a previsão do tempo?\" → fora_de_escopo\n"
    "- \"oi, tudo bem?\" → fora_de_escopo\n"
    "- \"me conta uma piada\" → fora_de_escopo"
)

# Resposta direta ao que a triagem barrou: mantém o usuário dentro do papel do
# assistente, sem inventar informação de RH nem acionar retrieval/tools.
SYSTEM_RESPOSTA_DIRETA = (
    "Você é um assistente de RH interno da empresa. A mensagem do usuário NÃO é "
    "sobre RH. Responda em português, de forma breve e cordial:\n"
    "- Se for uma saudação ou conversa fiada, cumprimente de volta e diga "
    "objetivamente no que você pode ajudar (políticas internas, saldo de férias "
    "e solicitação de férias).\n"
    "- Se for um assunto fora do escopo (ex.: previsão do tempo, piadas, "
    "conhecimento geral), explique gentilmente que você só trata de temas "
    "internos da empresa e ofereça ajuda com esses temas.\n"
    "Não invente informação. Responda apenas com a mensagem ao usuário."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        ("human", "Contexto:\n{contexto}\n\nPergunta: {pergunta}"),
    ]
)


class _Triagem(BaseModel):
    """Saída estruturada da triagem: um único campo com a categoria da pergunta."""

    categoria: Literal["rh", "fora_de_escopo"] = Field(
        description="'rh' se for assunto de RH da empresa; 'fora_de_escopo' caso contrário."
    )


# --- Modelo, tools e chains -------------------------------------------------

model = ChatAnthropic(model="claude-haiku-4-5", temperature=0)

# Decisão: modelo com as tools plugadas (decide se chama alguma).
TOOLS = [consultar_saldo_ferias, registrar_solicitacao_ferias]
TOOLS_POR_NOME = {t.name: t for t in TOOLS}
model_com_tools = model.bind_tools(TOOLS)

# Formatação: modelo que sempre devolve RespostaRH.
model_estruturado = model.with_structured_output(RespostaRH)

# Triagem: chamada leve que devolve só a categoria (rh / fora_de_escopo).
model_triagem = model.with_structured_output(_Triagem)

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

def triagem(state: EstadoRH) -> EstadoRH:
    """Classifica a pergunta em "rh" ou "fora_de_escopo" (chamada leve).

    Fail-open: qualquer falha na chamada classifica como "rh", para a triagem
    nunca derrubar uma pergunta legítima — na dúvida, segue o fluxo normal.
    """
    try:
        resultado = model_triagem.invoke(
            [
                SystemMessage(content=SYSTEM_TRIAGEM),
                HumanMessage(content=state["pergunta"]),
            ]
        )
        categoria = resultado.categoria
    except Exception:
        categoria = "rh"
    logger.info("[triagem] pergunta=%r categoria=%s", state["pergunta"], categoria)
    return {"categoria_triagem": categoria}


def rota_apos_triagem(state: EstadoRH) -> str:
    """Aresta condicional: fora de escopo responde direto; RH segue o fluxo."""
    return "fora_de_escopo" if state["categoria_triagem"] == "fora_de_escopo" else "rh"


def resposta_direta(state: EstadoRH) -> EstadoRH:
    """Responde com educação ao que a triagem barrou, sem retrieval nem tools.

    Devolve RespostaRH com fontes vazias e categoria "outro": a resposta não vem
    de uma política nem de um sistema interno.
    """
    msg = model.invoke(
        [
            SystemMessage(content=SYSTEM_RESPOSTA_DIRETA),
            HumanMessage(content=state["pergunta"]),
        ]
    )
    texto = (msg.content or "").strip()
    resposta = RespostaRH(
        resposta=texto,
        fontes=[],
        categoria="outro",
        confianca=1.0,
    )
    return {"resposta": resposta}


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
    """Recupera os chunks relevantes por similaridade para a consulta atual.

    Usa a consulta já definida no estado (a pergunta reescrita, quando o nó
    reescrever rodou numa tentativa anterior) ou, na primeira passada, a própria
    pergunta. A reescrita deixou de ser pré-processamento aqui: agora é reação a
    falha, feita no nó reescrever e realimentada neste nó pelo ciclo.
    """
    consulta = state.get("consulta") or state["pergunta"]
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


# Teto de tentativas de auto-correção: uma reescrita basta para tirar o ruído da
# pergunta; mais que isso vira loop improdutivo. O teto também garante que o
# ciclo reescrever → recuperar sempre termine.
MAX_TENTATIVAS = 1


def avaliar_resposta(state: EstadoRH) -> str:
    """Aresta condicional pós-geração: decide entre reescrever e encerrar.

    Só reescreve se a resposta não se sustentou na base (nenhuma fonte após a
    validação), a auto-correção está ligada e ainda há tentativa disponível.
    Caso contrário encerra — inclusive quando a base genuinamente não cobre o
    assunto, para não insistir nem alucinar.
    """
    sem_fontes = not state["resposta"].fontes
    if sem_fontes and state.get("auto_corrigir") and state.get("tentativas", 0) < MAX_TENTATIVAS:
        return "reescrever"
    return "fim"


def reescrever(state: EstadoRH) -> EstadoRH:
    """Reescreve a pergunta (remove ruído) e realimenta a busca — auto-correção.

    Normaliza a pergunta ORIGINAL (não a consulta anterior, para não acumular
    distorções) via _reescrever_pergunta, grava em consulta e incrementa o
    contador de tentativas, que serve de teto para o ciclo terminar.
    """
    tentativas = state.get("tentativas", 0) + 1
    consulta = _reescrever_pergunta(state["pergunta"])
    logger.info(
        "[auto-correcao] tentativa=%d original=%r reescrita=%r",
        tentativas, state["pergunta"], consulta,
    )
    return {"consulta": consulta, "tentativas": tentativas}


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
    g.add_node("triagem", triagem)
    g.add_node("resposta_direta", resposta_direta)
    g.add_node("decidir_rota", decidir_rota)
    g.add_node("recuperar", recuperar)
    g.add_node("gerar", gerar)
    g.add_node("validar_fontes", validar_fontes)
    g.add_node("reescrever", reescrever)
    g.add_node("executar_tools", executar_tools)
    g.add_node("formatar_tool", formatar_tool)

    # Triagem na entrada: fora de escopo responde direto; RH segue o fluxo.
    g.add_edge(START, "triagem")
    g.add_conditional_edges(
        "triagem",
        rota_apos_triagem,
        {"fora_de_escopo": "resposta_direta", "rh": "decidir_rota"},
    )
    g.add_edge("resposta_direta", END)
    g.add_conditional_edges(
        "decidir_rota",
        rota_apos_decisao,
        {"informativo": "recuperar", "tool": "executar_tools"},
    )
    # Rota informativa (política) com auto-correção: se a resposta não se
    # sustentou na base, reescrever normaliza a pergunta e realimenta recuperar
    # (ciclo com teto de tentativas); caso contrário, encerra.
    g.add_edge("recuperar", "gerar")
    g.add_edge("gerar", "validar_fontes")
    g.add_conditional_edges(
        "validar_fontes",
        avaliar_resposta,
        {"reescrever": "reescrever", "fim": END},
    )
    g.add_edge("reescrever", "recuperar")
    # Rota de tool.
    g.add_edge("executar_tools", "formatar_tool")
    g.add_edge("formatar_tool", END)

    return g.compile()


# Grafo compilado, pronto para invoke() (importado por app/chat.py).
grafo = _construir_grafo()
