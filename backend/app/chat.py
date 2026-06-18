"""
BRANCH DEMONSTRACAO — assistente FAKE para a turma ver o destino do curso.

NÃO usa LLM, NÃO faz embeddings, NÃO precisa de API key.

A "inteligência" aqui é matching por palavra-chave entre a pergunta e o
conteúdo de seis políticas de RH simuladas em backend/fake_data/. O objetivo
é simular o comportamento final do produto — pergunta livre, resposta
fundamentada com fonte citada — antes de termos LangChain de verdade.

Pipeline:
1. Carrega todos os .md de fake_data/ no startup.
2. Tokeniza a pergunta (lowercase, sem acento, sem stopwords PT-BR).
3. Pontua cada documento pela contagem de tokens da pergunta presentes em
   título + corpo.
4. Devolve o trecho mais relevante (primeiro parágrafo "útil" que cita um
   dos tokens) + arquivo + título como fonte.
5. Score zero → resposta padrão pedindo reformulação.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel


FAKE_DATA_DIR = Path(__file__).resolve().parent.parent / "fake_data"

STOPWORDS = {
    "a", "à", "as", "ao", "aos", "o", "os", "um", "uma", "uns", "umas",
    "de", "da", "do", "das", "dos", "em", "na", "no", "nas", "nos",
    "para", "por", "com", "sem", "sobre", "como", "que", "qual", "quais",
    "quanto", "quanta", "quantos", "quantas", "quando", "onde", "qual",
    "é", "ser", "está", "estão", "ter", "tenho", "tem", "temos", "têm",
    "tive", "tinha", "vou", "vai", "vamos", "ir", "posso", "pode", "podem",
    "se", "eu", "tu", "você", "vc", "vocês", "ele", "ela", "eles", "elas",
    "meu", "minha", "meus", "minhas", "seu", "sua", "seus", "suas",
    "isso", "isto", "esse", "essa", "este", "esta", "aquele", "aquela",
    "e", "ou", "mas", "porque", "pois", "então", "também", "já",
    "muito", "muita", "muitos", "muitas", "mais", "menos", "outro", "outra",
    "todo", "toda", "todos", "todas", "qualquer",
    "ai", "aí", "aqui", "lá", "agora", "hoje", "ontem", "amanhã",
}


def _normalize(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    no_accent = "".join(c for c in nfkd if not unicodedata.combining(c))
    return no_accent.lower()


def _tokens(text: str) -> set[str]:
    norm = _normalize(text)
    raw = re.findall(r"[a-z0-9]+", norm)
    return {t for t in raw if len(t) > 2 and t not in STOPWORDS}


@dataclass
class Secao:
    cabecalho: str
    corpo: str
    tokens_cabecalho: set[str]
    tokens_corpo: set[str]


@dataclass
class Doc:
    arquivo: str
    titulo: str
    conteudo: str
    tokens_filename: set[str]
    tokens_titulo: set[str]
    tokens_corpo: set[str]
    secoes: list[Secao]


def _seccionar(texto: str) -> list[Secao]:
    """Quebra o markdown em (cabeçalho ##, corpo até o próximo ##)."""
    linhas = texto.splitlines()
    secoes: list[Secao] = []
    cabecalho_atual = ""
    buffer: list[str] = []

    def flush():
        if not buffer and not cabecalho_atual:
            return
        corpo = "\n".join(buffer).strip()
        if not corpo and not cabecalho_atual:
            return
        secoes.append(
            Secao(
                cabecalho=cabecalho_atual,
                corpo=corpo,
                tokens_cabecalho=_tokens(cabecalho_atual),
                tokens_corpo=_tokens(corpo),
            )
        )

    for ln in linhas:
        if ln.startswith("## "):
            flush()
            cabecalho_atual = ln.lstrip("#").strip()
            buffer = []
        elif ln.startswith("# "):
            # h1 = título do doc; já capturado em Doc.titulo, não duplica no corpo.
            continue
        else:
            buffer.append(ln)
    flush()
    return secoes


def _carregar_docs() -> list[Doc]:
    docs: list[Doc] = []
    if not FAKE_DATA_DIR.exists():
        return docs
    for path in sorted(FAKE_DATA_DIR.glob("*.md")):
        texto = path.read_text(encoding="utf-8").strip()
        linhas = texto.splitlines()
        titulo = linhas[0].lstrip("#").strip() if linhas else path.stem
        docs.append(
            Doc(
                arquivo=path.name,
                titulo=titulo,
                conteudo=texto,
                tokens_filename=_tokens(path.stem),
                tokens_titulo=_tokens(titulo),
                tokens_corpo=_tokens(texto),
                secoes=_seccionar(texto),
            )
        )
    return docs


DOCS = _carregar_docs()


class ChatRequest(BaseModel):
    pergunta: str


class Fonte(BaseModel):
    arquivo: str
    titulo: str


class ChatResponse(BaseModel):
    resposta: str
    fontes: list[Fonte]


def _score_doc(doc: Doc, q: set[str]) -> int:
    # Filename e título pesam mais — o nome do arquivo carrega o tema do doc.
    return (
        len(doc.tokens_filename & q) * 5
        + len(doc.tokens_titulo & q) * 3
        + len(doc.tokens_corpo & q)
    )


def _melhor_secao(doc: Doc, q: set[str]) -> str:
    melhor = None
    melhor_score = -1
    for s in doc.secoes:
        # Cabeçalho pesa 3x: "Licença-paternidade" deve ganhar quando a pergunta
        # cita paternidade, mesmo que outro parágrafo também mencione.
        score = len(s.tokens_cabecalho & q) * 3 + len(s.tokens_corpo & q)
        if score > melhor_score:
            melhor_score = score
            melhor = s
    if not melhor:
        return doc.conteudo
    if melhor.cabecalho:
        return f"{melhor.cabecalho}\n\n{melhor.corpo}"
    return melhor.corpo


def responder(req: ChatRequest) -> ChatResponse:
    pergunta_tokens = _tokens(req.pergunta)
    if not DOCS or not pergunta_tokens:
        return ChatResponse(
            resposta=(
                "Não consegui entender a pergunta. Tente reformular usando "
                "palavras-chave como férias, home office, benefícios, reembolso, "
                "horário ou licenças."
            ),
            fontes=[],
        )

    ranking = sorted(DOCS, key=lambda d: _score_doc(d, pergunta_tokens), reverse=True)
    top = ranking[0]
    if _score_doc(top, pergunta_tokens) == 0:
        return ChatResponse(
            resposta=(
                "Não encontrei nenhuma política que cubra essa pergunta. Tópicos "
                "disponíveis: férias, home office, benefícios, reembolso, horário "
                "e licenças."
            ),
            fontes=[],
        )

    trecho = _melhor_secao(top, pergunta_tokens)
    return ChatResponse(
        resposta=trecho,
        fontes=[Fonte(arquivo=top.arquivo, titulo=top.titulo)],
    )
