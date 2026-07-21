"""
Fluxo de decisão do assistente de RH modelado como grafo (LangGraph).

O atendimento continua tendo dois caminhos que não cabem numa única chamada ao
modelo (tool calling e structured output se excluem — ver nota abaixo), mas
agora cada etapa é um nó explícito de um StateGraph, com o estado fluindo entre
eles em vez de variáveis locais. Uma triagem na entrada desvia o que está fora
do escopo antes de gastar retrieval/tools:

    START → triagem → (categoria?)
        ├── fora_de_escopo → resposta_direta → finalizar → END
        ├── conversacional → resposta_conversacional → finalizar → END
        └── rh → classificar_tipo → (tipo?)
                ├── politica → contextualizar → recuperar → gerar → validar_fontes → (sustentou?)
                │              ├── não, e auto_corrigir e < 1 retry → reescrever → recuperar
                │              └── sim, ou sem retry disponível → finalizar → END
                ├── tool → decidir_rota → executar_tools → formatar_tool → finalizar → END
                └── hibrida → iniciar_hibrido → [sub_politica ∥ sub_dados] → mesclar → finalizar → END

O estado é persistido por conversa (thread) por um checkpointer (ver
compilar_grafo): o campo `mensagens` acumula o histórico via add_messages e as
rotas convergem em finalizar, que registra o par (pergunta, resposta) do turno.

0. triagem: classifica a mensagem em "rh", "fora_de_escopo" ou "conversacional".
1. classificar_tipo: para perguntas de RH, distingue "tool" (dado de funcionário),
   "politica" (regra/norma) e "hibrida" (ambos). Na rota híbrida, decompõe a
   pergunta em duas sub-perguntas independentes (pergunta_dados, pergunta_politica).
2a. Rota política (contextualizar→recuperar→gerar→validar_fontes): RAG com
    auto-correção opcional (reescrever→recuperar, teto de 1 tentativa).
2b. Rota tool (decidir_rota→executar_tools→formatar_tool): tool calling.
2c. Rota híbrida (iniciar_hibrido→[sub_politica∥sub_dados]→mesclar): os dois
    ramos correm em paralelo (edges estáticas) e convergem em mesclar, que une
    as respostas via template. Auto-correção inativa no ramo híbrido.

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
import os
from typing import Annotated, Any, Literal, TypedDict

import numpy as np
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


def _trajetoria_turno(antigo: list[str] | None, novo: list[str] | None) -> list[str]:
    """Reducer da trajetória: acumula nós dentro do turno; None no novo = reset."""
    if novo is None:
        return []
    return (antigo or []) + novo
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

    A maioria das chaves usa sobrescrita padrão (cada nó devolve um dict parcial
    e o LangGraph substitui a chave). A exceção é `mensagens`, anotada com o
    reducer add_messages: ela ACUMULA o histórico da conversa entre turnos (o
    checkpointer persiste esse campo por thread), em vez de ser sobrescrita.
    Os campos cobrem a entrada (pergunta e flags de UI), o histórico e os
    valores intermediários que antes eram variáveis locais do fluxo.
    """

    # Entrada
    pergunta: str
    auto_corrigir: bool
    validar_teto: bool
    # Histórico da conversa (acumula via add_messages; persistido por thread)
    mensagens: Annotated[list, add_messages]
    # Intermediários
    categoria_triagem: str  # "rh", "fora_de_escopo" ou "conversacional"
    tipo_consulta: str  # "tool", "politica" ou "hibrida" (definido por classificar_tipo)
    consulta: str  # pergunta usada na busca/geração (reescrita, após auto-correção)
    tentativas: int  # nº de reescritas já feitas (teto para a auto-correção terminar)
    ai_msg: Any  # AIMessage da etapa de decisão (carrega os tool_calls)
    chunks: list[Document]  # chunks recuperados na rota informativa
    tool_messages: list[ToolMessage]  # resultados das tools na rota de tool
    # Rota híbrida: sub-perguntas decompostas e respostas parciais dos dois ramos
    pergunta_dados: str      # sub-pergunta sobre dado individual (rota híbrida)
    pergunta_politica: str   # sub-pergunta sobre regra/política (rota híbrida)
    resposta_politica: RespostaRH | None  # resultado do ramo de política
    resposta_dados: RespostaRH | None     # resultado do ramo de dados
    # Trajetória do turno (acumula por nó via reducer; zerada a cada turno via None)
    trajetoria: Annotated[list[str], _trajetoria_turno]
    # Avaliação de qualidade: similaridade cosseno entre a resposta e os chunks
    groundedness_score: float
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

# Contextualização da consulta: resolve referências ao histórico (pronomes,
# elipses) transformando a pergunta atual numa consulta autônoma. A busca
# (recuperar) não lê o histórico; sem este passo, um follow-up como "e ela pode
# tirar tudo de uma vez?" embutiria longe do chunk certo por não conter o
# assunto/entidade do turno anterior. Só age quando há histórico.
SYSTEM_CONTEXTUALIZAR = (
    "Dada a conversa anterior e a pergunta atual, reformule a pergunta atual "
    "numa consulta de busca AUTÔNOMA, que faça sentido sozinha sem o histórico.\n"
    "Regras:\n"
    "1. Resolva pronomes e referências (ela/ele/isso) usando o histórico — ex.: "
    "\"ela\" passa a ser o funcionário citado antes.\n"
    "2. PRESERVE O TIPO da pergunta: uma pergunta sobre REGRA/POLÍTICA continua "
    "uma consulta sobre a regra/política e NÃO deve virar uma consulta por dado "
    "individual (saldo, quantos dias fulano tem). Só reformule como consulta de "
    "saldo se a pergunta atual for realmente sobre o saldo de alguém.\n"
    "3. Mantenha o ASSUNTO do turno anterior explícito na consulta (ex.: a "
    "palavra \"férias\"), completando elipses.\n"
    "4. Não responda à pergunta. Responda SOMENTE com a consulta reformulada, "
    "sem explicação, aspas ou texto adicional.\n\n"
    "Exemplo:\n"
    "Histórico: usuário perguntou o saldo de férias da Ana; o assistente "
    "respondeu que a Ana tem 10 dias.\n"
    "Pergunta atual: \"E ela pode tirar tudo de uma vez?\"\n"
    "Consulta correta: \"regras da política de férias para tirar todos os dias "
    "de uma vez (fracionamento)\"\n"
    "Consulta ERRADA (não faça): \"saldo de férias da Ana para tirar de uma "
    "vez\" — vira dado individual e perde a regra."
)

# Triagem de entrada: classifica se a pergunta é assunto de RH ou está fora do
# escopo (saudações/small talk incluídos). Serve para desviar o que não é RH
# antes de gastar retrieval/tools, respondendo direto e com educação.
SYSTEM_TRIAGEM = (
    "Você é a triagem de um assistente de RH interno. Classifique a mensagem do "
    "usuário em UMA de três categorias:\n\n"
    "- \"rh\": qualquer pergunta cuja resposta envolva política, regra ou dado "
    "de funcionário — férias, home-office, benefícios, reembolso, horário, "
    "licenças, saldo de férias, solicitação de férias. É \"rh\" MESMO que o "
    "tema já tenha sido respondido neste thread e MESMO que seja repetição "
    "literal da pergunta anterior. Consulta se refaz na fonte, não na conversa.\n"
    "- \"fora_de_escopo\": saudações, conversa fiada (small talk) e temas "
    "claramente alheios ao trabalho (clima, piadas, esportes, notícias). "
    "Inclui cumprimentos com menção à conversa (ex.: \"oi, você se lembra do "
    "que falamos?\") — small talk prevalece sobre recall.\n"
    "- \"conversacional\": EXCLUSIVAMENTE perguntas sobre a própria conversa — "
    "o que foi dito, quem foi mencionado, como algo foi formulado. NÃO é "
    "\"conversacional\" se a resposta exigir acessar política, regra ou dado "
    "de funcionário, mesmo que o dado já tenha aparecido no histórico. "
    "Recall relembra a conversa; consulta se refaz na fonte.\n\n"
    "Regras de desempate:\n"
    "- Dúvida entre \"rh\" e \"conversacional\" → SEMPRE \"rh\".\n"
    "- Dúvida entre \"fora_de_escopo\" e \"conversacional\" → \"fora_de_escopo\".\n"
    "- Qualquer pergunta sobre temas de trabalho, empresa ou condições de "
    "trabalho é \"rh\", MESMO que o assunto pareça incomum.\n\n"
    "Exemplos — rh:\n"
    "- \"quero solicitar férias\" → rh\n"
    "- \"quantos dias a Ana tem?\" → rh\n"
    "- \"e quantos dias eu tenho direito a férias?\" → rh (regra de política)\n"
    "- \"posso trazer meu cachorro pro escritório?\" → rh\n"
    "- \"quantos dias a Ana tem?\" [perguntado de novo na mesma conversa] → rh "
    "(repetição literal de consulta de dado — refaz na fonte)\n"
    "- \"e ela pode tirar tudo de uma vez?\" [após resposta sobre saldo] → rh "
    "(pergunta sobre regra de política, não sobre o que foi dito)\n\n"
    "Exemplos — fora_de_escopo:\n"
    "- \"qual a previsão do tempo?\" → fora_de_escopo\n"
    "- \"oi, tudo bem?\" → fora_de_escopo\n"
    "- \"me conta uma piada\" → fora_de_escopo\n\n"
    "Exemplos — conversacional:\n"
    "- \"de quem estamos falando mesmo?\" → conversacional (só o nome, não dado)\n"
    "- \"qual era o nome da pessoa mesmo?\" → conversacional\n"
    "- \"você pode repetir o que disse antes?\" → conversacional\n"
    "- \"e quantos dias ela tinha mesmo?\" → conversacional (confirma o número "
    "JÁ DITO pelo assistente, não acessa o sistema)\n"
    "ATENÇÃO: \"e quantos dias ela tinha mesmo?\" é conversacional APENAS "
    "quando o assistente já respondeu o saldo nesta conversa e o usuário quer "
    "confirmação do número já dito. Se não houver resposta anterior com o saldo, "
    "é \"rh\"."
)

# Resposta conversacional: responde a partir do histórico da conversa, sem
# retrieval nem tools. Usado quando a triagem detecta recall explícito do turno
# anterior (ex.: "de quem estamos falando?", "qual era o saldo dela mesmo?").
SYSTEM_RESPOSTA_CONVERSACIONAL = (
    "Você é um assistente de RH interno da empresa. O usuário quer confirmar "
    "algo que o assistente já disse nesta conversa — um nome, um número ou "
    "uma informação presente no histórico acima.\n\n"
    "Tempo verbal: 'tinha', 'tem', 'era' e 'é' são variações do mesmo dado. "
    "Se o histórico registra 'X dias disponíveis' e o usuário pergunta quantos "
    "dias 'ela tinha', responda com o valor registrado ('X dias') — o tempo "
    "verbal não muda o dado, apenas o enquadramento da pergunta.\n\n"
    "REGRA DE EXTRAÇÃO: use APENAS o que está escrito no histórico. "
    "É PROIBIDO deduzir, calcular ou inferir números que não apareçam "
    "textualmente (ex.: não calcule total a partir de parcial, não subtraia "
    "nem some valores não mencionados). Copie o valor exato; não reinterprete.\n\n"
    "Se o dado pedido NÃO estiver no histórico, diga: "
    "'Essa informação não está na conversa. Poderia reformular a pergunta?' "
    "Não invente nem complete com conhecimento externo.\n\n"
    "Responda de forma breve e direta, sem ressalvas ('acredito que', "
    "'pelo que entendi'), sem repetir o turno anterior completo."
)

# Classificação do tipo de consulta de RH: distingue se exige tool (dado
# individual), retrieval (política) ou ambos (híbrida). Na rota híbrida o modelo
# também decompõe a pergunta em duas sub-perguntas independentes.
SYSTEM_CLASSIFICAR = (
    "Você classifica perguntas de RH em três tipos:\n\n"
    "- \"tool\": exige consultar um dado individual de um funcionário (saldo de "
    "férias, registro de solicitação de férias) — ex.: 'quantos dias a Ana tem?', "
    "'quero solicitar 5 dias de férias em julho'.\n"
    "- \"politica\": exige explicar uma regra ou política interna — ex.: 'quais "
    "os períodos permitidos?', 'como funciona o parcelamento de férias?', "
    "'quantos dias tenho direito?'.\n"
    "- \"hibrida\": exige TANTO consultar um dado individual QUANTO explicar uma "
    "regra — ex.: 'quantos dias a Ana tem e qual a regra para parcelar as férias?'.\n\n"
    "Quando tipo='hibrida', decomponha em duas sub-perguntas:\n"
    "- pergunta_dados: a parte sobre o dado individual (ex.: 'quantos dias de "
    "férias a Ana tem de saldo?').\n"
    "- pergunta_politica: a parte sobre a regra (ex.: 'qual a regra para "
    "parcelar as férias?').\n"
    "Quando tipo != 'hibrida', deixe pergunta_dados e pergunta_politica em branco.\n\n"
    "Regras de desempate:\n"
    "- Dúvida entre 'tool' e 'hibrida' → 'hibrida'.\n"
    "- Dúvida entre 'politica' e 'hibrida' → 'hibrida'.\n"
    "- Dúvida entre 'tool' e 'politica' → 'politica'."
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
        # Histórico da conversa antes da pergunta atual: permite resolver
        # referências a turnos anteriores (ex.: "ela", "e isso?") sem repetir
        # o contexto. Vazio no primeiro turno.
        MessagesPlaceholder("historico"),
        ("human", "Contexto:\n{contexto}\n\nPergunta: {pergunta}"),
    ]
)


class _Triagem(BaseModel):
    """Saída estruturada da triagem: um único campo com a categoria da pergunta."""

    categoria: Literal["rh", "fora_de_escopo", "conversacional"] = Field(
        description=(
            "'rh' se for assunto de RH da empresa; "
            "'fora_de_escopo' para saudações e small talk; "
            "'conversacional' para recall explícito do histórico da conversa."
        )
    )


class _TipoConsulta(BaseModel):
    """Saída estruturada do classificador de tipo de consulta RH."""

    tipo: Literal["tool", "politica", "hibrida"] = Field(
        description=(
            "'tool' para dado individual de funcionário; "
            "'politica' para regra/norma; "
            "'hibrida' quando exige ambos."
        )
    )
    pergunta_dados: str = Field(
        default="",
        description=(
            "Sub-pergunta sobre o dado individual. "
            "Preenchida apenas quando tipo='hibrida'."
        ),
    )
    pergunta_politica: str = Field(
        default="",
        description=(
            "Sub-pergunta sobre a regra ou política. "
            "Preenchida apenas quando tipo='hibrida'."
        ),
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

# Classificação do tipo de consulta RH: tool / politica / hibrida.
model_classificar = model.with_structured_output(_TipoConsulta)

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


# --- Avaliação de groundedness ----------------------------------------------

def _cosseno(a: list[float], b: list[float]) -> float:
    """Similaridade cosseno entre dois vetores.

    Retorna 0.0 se algum vetor for nulo (norma zero), evitando divisão por zero.
    """
    va, vb = np.array(a), np.array(b)
    norma = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / norma) if norma > 0 else 0.0




# --- Nós --------------------------------------------------------------------

def triagem(state: EstadoRH, config: RunnableConfig) -> EstadoRH:
    """Classifica a pergunta em "rh" ou "fora_de_escopo" (chamada leve).

    Fail-open: qualquer falha na chamada classifica como "rh", para a triagem
    nunca derrubar uma pergunta legítima — na dúvida, segue o fluxo normal.
    """
    # Ponto único por onde todo turno passa: registra o histórico que ENTRA
    # neste turno (cresce 2 msgs/turno, o par gravado por finalizar no anterior).
    historico = state.get("mensagens", [])
    thread = config.get("configurable", {}).get("thread_id", "")
    logger.info("[memoria] historico=%d mensagens thread=%s", len(historico), thread[:8])
    try:
        resultado = model_triagem.invoke(
            [
                SystemMessage(content=SYSTEM_TRIAGEM),
                *historico,
                HumanMessage(content=state["pergunta"]),
            ]
        )
        categoria = resultado.categoria
    except Exception:
        categoria = "rh"
    logger.info("[triagem] pergunta=%r categoria=%s", state["pergunta"], categoria)
    return {"categoria_triagem": categoria, "trajetoria": ["triagem"]}


def rota_apos_triagem(state: EstadoRH) -> str:
    """Aresta condicional: direciona para a rota correta após a triagem."""
    c = state["categoria_triagem"]
    if c == "fora_de_escopo":
        return "fora_de_escopo"
    if c == "conversacional":
        return "conversacional"
    return "rh"


def classificar_tipo(state: EstadoRH) -> EstadoRH:
    """Distingue 'tool', 'politica' ou 'hibrida' para perguntas de RH.

    Fail-open: qualquer falha na chamada classifica como 'politica', para não
    bloquear o fluxo — a rota de política é a mais conservadora.
    Na rota híbrida, o modelo decompõe a pergunta em duas sub-perguntas
    independentes (pergunta_dados, pergunta_politica).
    """
    try:
        resultado = model_classificar.invoke(
            [
                SystemMessage(content=SYSTEM_CLASSIFICAR),
                *state.get("mensagens", []),
                HumanMessage(content=state["pergunta"]),
            ]
        )
        tipo = resultado.tipo
        pergunta_dados = resultado.pergunta_dados or state["pergunta"]
        pergunta_politica = resultado.pergunta_politica or state["pergunta"]
    except Exception:
        tipo = "politica"
        pergunta_dados = state["pergunta"]
        pergunta_politica = state["pergunta"]
    logger.info(
        "[classificar_tipo] pergunta=%r tipo=%s", state["pergunta"], tipo
    )
    return {
        "tipo_consulta": tipo,
        "pergunta_dados": pergunta_dados,
        "pergunta_politica": pergunta_politica,
        "trajetoria": ["classificar_tipo"],
    }


def rota_apos_classificar(state: EstadoRH) -> str:
    """Aresta condicional: encaminha para tool, política ou fan-out híbrido."""
    return state.get("tipo_consulta", "politica")


def iniciar_hibrido(state: EstadoRH) -> EstadoRH:
    """Nó passagem que dispara os dois ramos paralelos via edges estáticas."""
    return {"trajetoria": ["iniciar_hibrido"]}


def sub_politica(state: EstadoRH) -> EstadoRH:
    """Ramo paralelo híbrido — RAG com a sub-pergunta de política.

    Executa o pipeline completo (contextualizar → recuperar → gerar →
    validar_fontes) inline, sem ciclo de auto-correção. Escreve em
    resposta_politica, não em resposta, para não conflitar com sub_dados.
    """
    pergunta = state.get("pergunta_politica") or state["pergunta"]
    mensagens = state.get("mensagens", [])
    # Contextualizar
    if mensagens:
        try:
            msg = model.invoke(
                [
                    SystemMessage(content=SYSTEM_CONTEXTUALIZAR),
                    *mensagens,
                    HumanMessage(content=pergunta),
                ]
            )
            consulta = (msg.content or "").strip() or pergunta
        except Exception:
            consulta = pergunta
    else:
        consulta = pergunta
    logger.info("[sub_politica] consulta=%r", consulta)
    # Recuperar
    chunks = buscar(consulta, k=TOP_K)
    # Gerar
    resposta = chain.invoke(
        {
            "contexto": _formatar_contexto(chunks),
            "pergunta": consulta,
            "historico": mensagens,
        }
    )
    # Validar fontes
    resposta.fontes = _validar_fontes(resposta.fontes, chunks)
    logger.info(
        "[sub_politica] fontes=%s", [f.arquivo for f in resposta.fontes]
    )
    return {"resposta_politica": resposta, "trajetoria": ["sub_politica"]}


def sub_dados(state: EstadoRH) -> EstadoRH:
    """Ramo paralelo híbrido — tool calling com a sub-pergunta de dados.

    Executa decide → tool(s) → formata inline. Escreve em resposta_dados,
    não em resposta, para não conflitar com sub_politica.
    """
    pergunta = state.get("pergunta_dados") or state["pergunta"]
    mensagens = state.get("mensagens", [])
    ai_msg = model_com_tools.invoke(
        [
            SystemMessage(content=SYSTEM_DECISAO),
            *mensagens,
            HumanMessage(content=pergunta),
        ]
    )
    if not ai_msg.tool_calls:
        logger.info("[sub_dados] sem tool_calls para %r", pergunta)
        return {
            "resposta_dados": RespostaRH(
                resposta="Não foi possível consultar os dados.",
                fontes=[],
                categoria="outro",
                confianca=0.0,
            ),
            "trajetoria": ["sub_dados"],
        }
    tool_messages: list[ToolMessage] = []
    for call in ai_msg.tool_calls:
        tool = TOOLS_POR_NOME.get(call["name"])
        resultado = (
            tool.invoke(call["args"]) if tool else "Ferramenta desconhecida"
        )
        if state.get("validar_teto") and call["name"] == "consultar_saldo_ferias":
            alerta = alerta_teto(call["args"].get("funcionario", ""))
            if alerta:
                resultado = f"{resultado} {alerta}"
        tool_messages.append(
            ToolMessage(content=str(resultado), tool_call_id=call["id"])
        )
    resposta = model_estruturado.invoke(
        [
            SystemMessage(content=SYSTEM_FORMATA_TOOL),
            HumanMessage(content=pergunta),
            ai_msg,
            *tool_messages,
        ]
    )
    logger.info("[sub_dados] categoria=%s confianca=%s", resposta.categoria, resposta.confianca)
    return {"resposta_dados": resposta, "trajetoria": ["sub_dados"]}


def mesclar(state: EstadoRH) -> EstadoRH:
    """Nó de junção: une as respostas dos dois ramos paralelos via template.

    Categoria herdada da resposta de política (fallback 'outro').
    Confiança = menor das duas respostas (mais conservador).
    Fontes = somente as da política (dados vêm de sistema interno).
    """
    pol = state.get("resposta_politica")
    dad = state.get("resposta_dados")
    partes = []
    if dad and dad.resposta:
        partes.append(f"**Dados consultados:**\n{dad.resposta}")
    if pol and pol.resposta:
        partes.append(f"**Regras da política:**\n{pol.resposta}")
    texto = "\n\n".join(partes) if partes else "Não foi possível obter as informações."
    categoria = (pol.categoria if pol else None) or (dad.categoria if dad else None) or "outro"
    confianças = [r.confianca for r in [pol, dad] if r is not None]
    confianca = min(confianças) if confianças else 0.0
    fontes = pol.fontes if pol else []
    return {
        "resposta": RespostaRH(
            resposta=texto,
            fontes=fontes,
            categoria=categoria,
            confianca=confianca,
        ),
        "trajetoria": ["mesclar"],
    }


_SEM_HISTORICO = (
    "Não encontrei contexto anterior nesta conversa. "
    "Poderia reformular a pergunta de forma completa?"
)


def resposta_conversacional(state: EstadoRH) -> EstadoRH:
    """Responde a perguntas de recall do histórico, sem retrieval nem tools.

    Análogo a resposta_direta, mas o contexto é o histórico da conversa: o
    modelo responde apenas com base no que já foi dito neste thread.
    Se não houver histórico, retorna resposta fixa sem chamar o modelo.
    Devolve RespostaRH com fontes vazias e categoria "outro".
    """
    mensagens = state.get("mensagens", [])
    logger.info("[conversacional] historico=%d mensagens", len(mensagens))
    if not mensagens:
        return {
            "resposta": RespostaRH(
                resposta=_SEM_HISTORICO,
                fontes=[],
                categoria="outro",
                confianca=1.0,
            ),
            "trajetoria": ["resposta_conversacional"],
        }
    msg = model.invoke(
        [
            SystemMessage(content=SYSTEM_RESPOSTA_CONVERSACIONAL),
            *mensagens,
            HumanMessage(content=state["pergunta"]),
        ]
    )
    texto = (msg.content or "").strip()
    return {
        "resposta": RespostaRH(
            resposta=texto,
            fontes=[],
            categoria="outro",
            confianca=1.0,
        ),
        "trajetoria": ["resposta_conversacional"],
    }


def resposta_direta(state: EstadoRH) -> EstadoRH:
    """Responde com educação ao que a triagem barrou, sem retrieval nem tools.

    Devolve RespostaRH com fontes vazias e categoria "outro": a resposta não vem
    de uma política nem de um sistema interno.
    """
    msg = model.invoke(
        [
            SystemMessage(content=SYSTEM_RESPOSTA_DIRETA),
            *state.get("mensagens", []),
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
    return {"resposta": resposta, "trajetoria": ["resposta_direta"]}


def decidir_rota(state: EstadoRH) -> EstadoRH:
    """Etapa de decisão: o modelo (com tools plugadas) decide se usa alguma.

    O AIMessage resultante segue no estado: seus tool_calls definem a rota e,
    na rota de tool, ele é reinjetado na formatação final.
    """
    ai_msg = model_com_tools.invoke(
        [
            SystemMessage(content=SYSTEM_DECISAO),
            *state.get("mensagens", []),
            HumanMessage(content=state["pergunta"]),
        ]
    )
    return {"ai_msg": ai_msg, "trajetoria": ["decidir_rota"]}


def rota_apos_decisao(state: EstadoRH) -> str:
    """Aresta condicional: encaminha para a rota de tool ou a informativa.

    Sem tool_calls → pergunta de política (rota informativa). Com tool(s) → o
    código executa cada uma (rota de tool).
    """
    return "tool" if state["ai_msg"].tool_calls else "informativo"


def contextualizar(state: EstadoRH) -> EstadoRH:
    """Reformula a pergunta atual em consulta autônoma resolvendo o histórico.

    Roda na entrada da rota informativa, antes da busca: resolve pronomes e
    elipses (ex.: "ela" → "Ana", "tirar tudo de uma vez" → "tirar todos os dias
    de férias de uma vez") para a consulta não depender do turno anterior — a
    busca (recuperar) é cega ao histórico. No primeiro turno (sem histórico) a
    consulta é a própria pergunta. Fail-open: qualquer falha cai para a pergunta
    original, pois a contextualização reforça a busca e não pode degradá-la.
    """
    mensagens = state.get("mensagens", [])
    if not mensagens:
        return {"consulta": state["pergunta"], "trajetoria": ["contextualizar"]}
    try:
        msg = model.invoke(
            [
                SystemMessage(content=SYSTEM_CONTEXTUALIZAR),
                *mensagens,
                HumanMessage(content=state["pergunta"]),
            ]
        )
        consulta = (msg.content or "").strip() or state["pergunta"]
    except Exception:
        consulta = state["pergunta"]
    logger.info(
        "[contextualizar] original=%r consulta=%r", state["pergunta"], consulta
    )
    return {"consulta": consulta, "trajetoria": ["contextualizar"]}


def recuperar(state: EstadoRH) -> EstadoRH:
    """Recupera os chunks relevantes por similaridade para a consulta atual.

    Usa a consulta já definida no estado (a pergunta reescrita, quando o nó
    reescrever rodou numa tentativa anterior) ou, na primeira passada, a própria
    pergunta. A reescrita deixou de ser pré-processamento aqui: agora é reação a
    falha, feita no nó reescrever e realimentada neste nó pelo ciclo.
    """
    consulta = state.get("consulta") or state["pergunta"]
    chunks = buscar(consulta, k=TOP_K)
    return {"consulta": consulta, "chunks": chunks, "trajetoria": ["recuperar"]}


def gerar(state: EstadoRH) -> EstadoRH:
    """Formula a resposta de política só com base nos chunks recuperados."""
    resposta = chain.invoke(
        {
            "contexto": _formatar_contexto(state["chunks"]),
            "pergunta": state["consulta"],
            "historico": state.get("mensagens", []),
        }
    )
    return {"resposta": resposta, "trajetoria": ["gerar"]}


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
    return {"resposta": resposta, "trajetoria": ["validar_fontes"]}


def avaliar_groundedness(state: EstadoRH, config: RunnableConfig) -> dict:
    """Calcula a similaridade cosseno entre a resposta gerada e os chunks recuperados.

    Embute a resposta e cada chunk, e computa o score como o máximo das
    similaridades cosseno (resposta vs. cada chunk). Um score alto indica que
    a resposta está ancorada no material recuperado; baixo sugere deriva ou
    resposta genérica.

    Só executa para a rota de política (RAG): sem resposta ou sem chunks o
    score é 0.0 (ex.: rota tool, fora-de-escopo, conversacional). O score é
    logado no LangFuse como métrica nomeada "groundedness" quando o client
    estiver configurado.
    """
    resposta_obj = state.get("resposta")
    chunks = state.get("chunks", [])
    resposta_texto = resposta_obj.resposta if resposta_obj else ""

    if not resposta_texto or not chunks:
        return {"groundedness_score": 0.0, "trajetoria": ["avaliar_groundedness"]}

    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        textos_chunks = [c.page_content for c in chunks]
        vecs = embeddings.embed_documents([resposta_texto] + textos_chunks)
        vec_resposta = vecs[0]
        scores = [_cosseno(vec_resposta, v) for v in vecs[1:]]
        score = round(max(scores), 4)
    except Exception as exc:
        logger.warning("[groundedness] falha ao calcular score: %s", exc)
        return {"groundedness_score": 0.0, "trajetoria": ["avaliar_groundedness"]}

    logger.info("[groundedness] score=%.4f pergunta=%r", score, state.get("pergunta", ""))
    return {"groundedness_score": score, "trajetoria": ["avaliar_groundedness"]}


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
    return {"consulta": consulta, "tentativas": tentativas, "trajetoria": ["reescrever"]}


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
    return {"tool_messages": tool_messages, "trajetoria": ["executar_tools"]}


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
    return {"resposta": resposta, "trajetoria": ["formatar_tool"]}


def finalizar(state: EstadoRH) -> EstadoRH:
    """Nó terminal: registra o par (pergunta, resposta) no histórico da thread.

    Ponto ÚNICO de escrita no histórico, comum às três rotas (fora de escopo,
    informativa e de tool). Roda uma vez por turno, depois que a resposta final
    já está no estado — evita o registro duplicado que ocorreria se cada rota
    (ou o nó validar_fontes, que pode repetir na auto-correção) gravasse por si.

    Só grava o TEXTO das mensagens (HumanMessage/AIMessage de conteúdo puro),
    não o AIMessage de decisão com tool_calls: manter tool_calls no histórico
    quebraria uma reinvocação futura de bind_tools (tool_call sem ToolMessage
    correspondente). O add_messages acumula esse par ao histórico persistido.
    """
    resposta = state["resposta"]
    return {
        "mensagens": [
            HumanMessage(content=state["pergunta"]),
            AIMessage(content=resposta.resposta),
        ],
        "trajetoria": ["finalizar"],
    }


# --- Montagem do grafo ------------------------------------------------------

def montar_grafo() -> StateGraph:
    """Monta o StateGraph do fluxo de decisão (nós + arestas), sem compilar.

    A montagem é mantida separada da compilação para o checkpointer poder ser
    injetado no momento certo (aberto no lifespan do FastAPI) via compilar_grafo.
    """
    g = StateGraph(EstadoRH)
    g.add_node("triagem", triagem)
    g.add_node("classificar_tipo", classificar_tipo)
    g.add_node("resposta_conversacional", resposta_conversacional)
    g.add_node("resposta_direta", resposta_direta)
    g.add_node("decidir_rota", decidir_rota)
    g.add_node("contextualizar", contextualizar)
    g.add_node("recuperar", recuperar)
    g.add_node("gerar", gerar)
    g.add_node("validar_fontes", validar_fontes)
    g.add_node("avaliar_groundedness", avaliar_groundedness)
    g.add_node("reescrever", reescrever)
    g.add_node("executar_tools", executar_tools)
    g.add_node("formatar_tool", formatar_tool)
    g.add_node("iniciar_hibrido", iniciar_hibrido)
    g.add_node("sub_politica", sub_politica)
    g.add_node("sub_dados", sub_dados)
    g.add_node("mesclar", mesclar)
    g.add_node("finalizar", finalizar)

    # Triagem na entrada: fora de escopo/conversacional respondem diretamente;
    # RH passa pelo classificador de tipo antes de seguir o fluxo.
    g.add_edge(START, "triagem")
    g.add_conditional_edges(
        "triagem",
        rota_apos_triagem,
        {
            "fora_de_escopo": "resposta_direta",
            "conversacional": "resposta_conversacional",
            "rh": "classificar_tipo",
        },
    )
    g.add_edge("resposta_conversacional", "finalizar")
    g.add_edge("resposta_direta", "finalizar")
    # Classificador: ramifica para política, tool ou fan-out híbrido.
    g.add_conditional_edges(
        "classificar_tipo",
        rota_apos_classificar,
        {
            "politica": "contextualizar",
            "tool": "decidir_rota",
            "hibrida": "iniciar_hibrido",
        },
    )
    # Rota de tool pura.
    g.add_conditional_edges(
        "decidir_rota",
        rota_apos_decisao,
        {"informativo": "contextualizar", "tool": "executar_tools"},
    )
    g.add_edge("executar_tools", "formatar_tool")
    g.add_edge("formatar_tool", "finalizar")
    # Rota política (RAG): com ciclo de auto-correção opcional.
    # validar_fontes → avaliar_groundedness → (condicional: reescrever | finalizar)
    # O score de groundedness é calculado após a validação de fontes, quando os
    # chunks e a resposta final do turno já estão consolidados no estado.
    g.add_edge("contextualizar", "recuperar")
    g.add_edge("recuperar", "gerar")
    g.add_edge("gerar", "validar_fontes")
    g.add_edge("validar_fontes", "avaliar_groundedness")
    g.add_conditional_edges(
        "avaliar_groundedness",
        avaliar_resposta,
        {"reescrever": "reescrever", "fim": "finalizar"},
    )
    g.add_edge("reescrever", "recuperar")
    # Rota híbrida: fan-out paralelo via edges estáticas + barreira de junção.
    g.add_edge("iniciar_hibrido", "sub_politica")
    g.add_edge("iniciar_hibrido", "sub_dados")
    g.add_edge(["sub_politica", "sub_dados"], "mesclar")
    g.add_edge("mesclar", "finalizar")
    # Nó terminal comum: registra o histórico e encerra o turno.
    g.add_edge("finalizar", END)

    return g


def compilar_grafo(checkpointer=None):
    """Compila o grafo do fluxo, opcionalmente com um checkpointer.

    Com um checkpointer (PostgresSaver), o estado passa a ser persistido por
    thread (config["configurable"]["thread_id"]), dando memória de conversa
    entre turnos. Sem ele, o grafo roda sem memória (comportamento anterior).
    """
    return montar_grafo().compile(checkpointer=checkpointer)
