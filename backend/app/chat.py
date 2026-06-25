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

O contexto das políticas é o conteúdo dos .md em backend/fake_data/, injetado
direto no prompt (são poucos documentos e cabem no contexto).

Requer ANTHROPIC_API_KEY no ambiente (ver .env / docker-compose.yml).
"""
from __future__ import annotations

from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.schemas import Fonte, RespostaRH
from app.tools import (
    alerta_teto,
    consultar_saldo_ferias,
    registrar_solicitacao_ferias,
)


FAKE_DATA_DIR = Path(__file__).resolve().parent.parent / "fake_data"


# --- Contrato de entrada da API ---------------------------------------------

class ChatRequest(BaseModel):
    pergunta: str
    # Quando ligado, o saldo consultado é validado contra o teto da política
    # (a tool sinaliza inconsistências). Controlado pela UI.
    validar_teto: bool = False


# --- Carregamento dos documentos (stuffing) ---------------------------------

def _carregar_docs() -> list[tuple[str, str, str]]:
    """Devolve [(arquivo, titulo, conteudo), ...] para os .md de fake_data/."""
    docs: list[tuple[str, str, str]] = []
    if not FAKE_DATA_DIR.exists():
        return docs
    for path in sorted(FAKE_DATA_DIR.glob("*.md")):
        texto = path.read_text(encoding="utf-8").strip()
        primeira = texto.splitlines()[0] if texto else path.stem
        titulo = primeira.lstrip("#").strip() or path.stem
        docs.append((path.name, titulo, texto))
    return docs


DOCS = _carregar_docs()

# Contexto único com delimitadores claros, para o modelo conseguir citar a
# política de origem de cada informação.
CONTEXTO = "\n\n---\n\n".join(
    f"[Política: {titulo} | arquivo: {arquivo}]\n{conteudo}"
    for arquivo, titulo, conteudo in DOCS
)

# Documentos disponíveis no contexto (por ora, os 6).
FONTES = [Fonte(arquivo=arquivo, titulo=titulo) for arquivo, titulo, _ in DOCS]


# --- Modelo, tools e chains -------------------------------------------------

SYSTEM = (
    "Você é um assistente de RH. Responda de forma clara e objetiva, SOMENTE "
    "com base no contexto fornecido. Se a resposta não estiver no contexto, "
    "diga que não encontrou e sugira procurar o RH. Sempre cite de qual "
    "política veio a informação.\n\n"
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

        # Sem tool → pergunta de política: fluxo de structured output existente.
        if not ai_msg.tool_calls:
            return chain.invoke({"contexto": CONTEXTO, "pergunta": req.pergunta})

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
