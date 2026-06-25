"""
Chain de RH: prompt | model com saída estruturada.

Responde a perguntas com base nas políticas internas de RH (LCEL):

- prompt: ChatPromptTemplate com um system fixo (persona de RH + regra de
  só responder com base no contexto) e um human com {contexto} e {pergunta}.
- model:  ChatAnthropic, claude-haiku-4-5, temperatura 0, com saída
  estruturada no schema RespostaRH (resposta, fontes, categoria, confianca).

O contexto é o conteúdo das políticas em backend/fake_data/, injetado direto
no prompt (são poucos documentos e cabem no contexto).

Requer ANTHROPIC_API_KEY no ambiente (ver .env / docker-compose.yml).
"""
from __future__ import annotations

from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.schemas import Fonte, RespostaRH


FAKE_DATA_DIR = Path(__file__).resolve().parent.parent / "fake_data"


# --- Contrato de entrada da API ---------------------------------------------

class ChatRequest(BaseModel):
    pergunta: str


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


# --- Chain LCEL: prompt | model com saída estruturada -----------------------

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

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        ("human", "Contexto:\n{contexto}\n\nPergunta: {pergunta}"),
    ]
)

model = ChatAnthropic(model="claude-haiku-4-5", temperature=0)

chain = prompt | model.with_structured_output(RespostaRH)


# --- Rota -------------------------------------------------------------------

def responder(req: ChatRequest) -> RespostaRH:
    try:
        return chain.invoke({"contexto": CONTEXTO, "pergunta": req.pergunta})
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
