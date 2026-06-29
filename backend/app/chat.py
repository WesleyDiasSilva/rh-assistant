"""
Chain de RH com tools e saída estruturada.

A rota /api/chat funciona em duas etapas, porque tool calling e structured
output não cabem na mesma chamada (ver nota abaixo):

1. Decisão/execução (bind_tools): o modelo recebe a pergunta com as tools
   plugadas e decide. Se não pedir tool, é pergunta de política. Se pedir
   tool(s), o código as executa (lendo name/args de tool_calls) e devolve o
   resultado ao modelo via ToolMessage.
2. Formatação (with_structured_output): o modelo formula a resposta final
   sempre no schema RespostaRH (resposta, fontes, categoria, confianca), para
   o contrato da API ser o mesmo nos dois caminhos.

Nota sobre o encontro tool calling + structured output: with_structured_output
já é implementado forçando uma tool call para o schema RespostaRH. Isso conflita
com bind_tools na mesma chamada — o modelo seria forçado a "responder" e não
poderia decidir por uma tool de negócio. Por isso separamos em duas etapas: a
primeira decide/executa a tool; a segunda formata em RespostaRH.

No ramo informativo, o contexto vem de retrieval: busca por similaridade na
base vetorial (ver app/retrieval.py) traz só os chunks relevantes à pergunta,
em vez de injetar todas as políticas no prompt.

Requer ANTHROPIC_API_KEY (geração) e OPENAI_API_KEY (embeddings da busca) no
ambiente (ver .env / docker-compose.yml).
"""
from __future__ import annotations

import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.retrieval import TOP_K, buscar
from app.schemas import Fonte, RespostaRH
from app.tools import (
    alerta_teto,
    consultar_saldo_ferias,
    registrar_solicitacao_ferias,
)

logger = logging.getLogger(__name__)


# --- Contrato de entrada da API ---------------------------------------------

class ChatRequest(BaseModel):
    pergunta: str
    # Quando ligado, o saldo consultado é validado contra o teto da política
    # (a tool sinaliza inconsistências). Controlado pela UI.
    validar_teto: bool = False


# --- Recuperação de contexto (retrieval) ------------------------------------

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


# --- Modelo, tools e chains -------------------------------------------------

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

# Etapa 1: system que orienta o modelo a decidir entre tool e política.
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

# Etapa 2 (caminho com tool): system que orienta a formatar a resposta final
# a partir do resultado da(s) ferramenta(s).
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

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        ("human", "Contexto:\n{contexto}\n\nPergunta: {pergunta}"),
    ]
)

model = ChatAnthropic(model="claude-haiku-4-5", temperature=0)

# Etapa 1: modelo com as tools plugadas (decide se chama alguma).
TOOLS = [consultar_saldo_ferias, registrar_solicitacao_ferias]
TOOLS_POR_NOME = {t.name: t for t in TOOLS}
model_com_tools = model.bind_tools(TOOLS)

# Etapa 2: modelo que sempre devolve RespostaRH.
model_estruturado = model.with_structured_output(RespostaRH)

# Chain de política (sem tool): prompt com contexto | saída estruturada.
chain = prompt | model_estruturado


# --- Rota -------------------------------------------------------------------

def responder(req: ChatRequest) -> RespostaRH:
    try:
        # Etapa 1: o modelo decide se usa alguma tool.
        ai_msg = model_com_tools.invoke(
            [
                SystemMessage(content=SYSTEM_DECISAO),
                HumanMessage(content=req.pergunta),
            ]
        )

        # Sem tool → pergunta de política: recupera os chunks relevantes e
        # responde só com base neles (retrieval em vez de stuffar tudo).
        if not ai_msg.tool_calls:
            chunks = buscar(req.pergunta, k=TOP_K)
            resposta = chain.invoke(
                {"contexto": _formatar_contexto(chunks), "pergunta": req.pergunta}
            )
            # As fontes citadas vêm do modelo (que sabe quais chunks usou); aqui
            # só validamos contra o conjunto recuperado. Groundedness preservado:
            # se não encontrou na base, o modelo deixa fontes vazias e a validação
            # mantém vazio.
            resposta.fontes = _validar_fontes(resposta.fontes, chunks)
            # Log de inspeção do retrieval: o conjunto recuperado traz os k chunks
            # (com ruído, de propósito), mas as fontes citadas devem ser um
            # subconjunto fiel. Registrar ambos permite observar essa diferença.
            recuperados = sorted(
                {c.metadata.get("arquivo") for c in chunks if c.metadata.get("arquivo")}
            )
            citados = [f.arquivo for f in resposta.fontes]
            logger.info(
                "retrieval informativo: pergunta=%r recuperados=%s citados=%s",
                req.pergunta, recuperados, citados,
            )
            return resposta

        # Com tool(s): o código executa cada uma e devolve via ToolMessage.
        tool_messages: list[ToolMessage] = []
        for call in ai_msg.tool_calls:
            tool = TOOLS_POR_NOME.get(call["name"])
            resultado = (
                tool.invoke(call["args"]) if tool else "Ferramenta desconhecida"
            )
            # Validação opcional do teto da política (controlada pela UI):
            # o sistema sinaliza saldo acima do máximo, sem depender do modelo.
            if req.validar_teto and call["name"] == "consultar_saldo_ferias":
                alerta = alerta_teto(call["args"].get("funcionario", ""))
                if alerta:
                    resultado = f"{resultado} {alerta}"
            tool_messages.append(
                ToolMessage(content=str(resultado), tool_call_id=call["id"])
            )

        # Etapa 2: o modelo formula a resposta final como RespostaRH.
        return model_estruturado.invoke(
            [
                SystemMessage(content=SYSTEM_FORMATA_TOOL),
                HumanMessage(content=req.pergunta),
                ai_msg,
                *tool_messages,
            ]
        )
    except Exception as exc:  # ex.: sem ANTHROPIC_API_KEY, falha de rede/API
        return RespostaRH(
            resposta=(
                "Não consegui consultar o assistente agora. Verifique se a "
                "ANTHROPIC_API_KEY está configurada e tente novamente. "
                f"(detalhe: {exc})"
            ),
            fontes=[],
            categoria="outro",
            confianca=0.0,
        )
