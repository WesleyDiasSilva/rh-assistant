"""Infraestrutura de RAG (retrieval) sobre as políticas de RH.

Centraliza embeddings, splitting, a base vetorial (PGVector) e as operações de
indexação e busca, para que o restante da aplicação só consuma estas funções.

Provedores distintos para cada etapa: a OpenAI gera os embeddings
(text-embedding-3-small) e a Anthropic gera as respostas no chat. São papéis
independentes — embeddar texto e gerar linguagem — e cada provedor é usado onde
tem o melhor custo/qualidade.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

import psycopg

from app import db

logger = logging.getLogger(__name__)

# Pasta com os documentos de política (.md) que alimentam a base vetorial.
FAKE_DATA_DIR = Path(__file__).resolve().parent.parent / "fake_data"

# Modelo de embeddings. O "small" é barato e suficiente para documentos curtos
# de política; o "large" só se justificaria com acervo grande e buscas sutis.
EMBEDDING_MODEL = "text-embedding-3-small"

# Tamanho do chunk em caracteres. Trade-off: chunks grandes preservam contexto
# mas diluem a relevância (a similaridade fica "borrada" e o top-k traz texto
# irrelevante junto); chunks pequenos são mais precisos mas fragmentam a ideia.
# As políticas aqui são curtas (poucos parágrafos), então 800 mantém uma seção
# coesa por chunk sem estourar contexto.
CHUNK_SIZE = 800

# Sobreposição entre chunks consecutivos, para uma frase cortada na fronteira
# ainda aparecer inteira em um dos chunks. ~12% do chunk equilibra recall e
# duplicação de conteúdo.
CHUNK_OVERLAP = 100

# Nome lógico do conjunto de vetores no PGVector. A "collection" agrupa os
# embeddings de um mesmo domínio (aqui, as políticas), isolando-os de eventuais
# outras coleções na mesma base.
COLLECTION_NAME = "politicas"

# Quantos chunks a busca retorna. Trade-off do top-k: poucos chunks podem perder
# a passagem certa (recall baixo); muitos trazem ruído e gastam contexto/tokens
# na geração. 4 cobre bem perguntas que cruzam 1-2 seções de política.
#
# Configurável por ambiente (TOP_K) para permitir variar o parâmetro sem editar
# código — útil para medir o efeito do top-k sobre a qualidade das respostas. O
# default preserva o comportamento anterior.
TOP_K = int(os.getenv("TOP_K", "4"))

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def get_embeddings() -> OpenAIEmbeddings:
    """Cliente de embeddings da OpenAI usado para indexar e para buscar."""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def get_vectorstore() -> PGVector:
    """Retorna o PGVector configurado para a collection de políticas.

    Usa a URL SQLAlchemy (psycopg3) de db.get_sqlalchemy_url(); use_jsonb=True
    armazena os metadados em jsonb (mais eficiente para filtros/consulta).
    """
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=db.get_sqlalchemy_url(),
        use_jsonb=True,
    )


def indexar_documento(texto: str, metadados: dict) -> int:
    """Faz split do texto em chunks e os adiciona à base vetorial.

    Os metadados são propagados para cada chunk e devem incluir ao menos
    'arquivo' e 'titulo' — é deles que as fontes citadas na resposta saem depois.

    IDs estáveis (arquivo + índice do chunk) tornam a indexação idempotente:
    reexecutar com o mesmo conteúdo faz upsert sobre os mesmos IDs, sem duplicar.

    Retorna quantos chunks foram criados.
    """
    arquivo = metadados.get("arquivo")
    if not arquivo:
        raise ValueError("metadados deve incluir 'arquivo' para gerar IDs estáveis")

    # Logs por etapa do pipeline (split -> embed -> store), para inspeção do
    # comportamento da indexação ao acompanhar o backend.
    pedacos = _splitter.split_text(texto)
    logger.info(
        "[indexação] split: arquivo=%s %d chars -> %d chunks",
        arquivo, len(texto), len(pedacos),
    )
    documentos = [
        Document(page_content=pedaco, metadata={**metadados, "chunk": i})
        for i, pedaco in enumerate(pedacos)
    ]
    ids = [f"{arquivo}::chunk-{i}" for i in range(len(documentos))]

    if documentos:
        logger.info(
            "[indexação] embeddings: modelo=%s chunks=%d", EMBEDDING_MODEL, len(documentos)
        )
        get_vectorstore().add_documents(documentos, ids=ids)
        logger.info(
            "[indexação] gravado: %d chunks na collection '%s'",
            len(documentos), COLLECTION_NAME,
        )
    return len(documentos)


def buscar(pergunta: str, k: int = TOP_K) -> list[Document]:
    """Busca por similaridade e retorna os k chunks mais próximos da pergunta.

    Usa similarity_search_with_score para obter, além dos chunks, o score de
    similaridade de cada um (distância no espaço vetorial: menor = mais próximo).
    O score é anexado em metadata['score'] de cada Document, preservando o tipo
    de retorno (list[Document]): os chamadores existentes seguem inalterados e
    quem precisar do score lê doc.metadata['score'].
    """
    resultados = get_vectorstore().similarity_search_with_score(pergunta, k=k)
    chunks: list[Document] = []
    for posicao, (doc, score) in enumerate(resultados, start=1):
        doc.metadata["score"] = score
        chunks.append(doc)
        # Log por chunk recuperado (posição + arquivo + score), para inspeção do
        # ranking da busca — o score evidencia quão próximo cada chunk ficou.
        logger.info(
            "[busca] %dº %s score=%.4f",
            posicao, doc.metadata.get("arquivo"), score,
        )
    return chunks


def carregar_docs() -> list[tuple[str, str, str]]:
    """Devolve [(arquivo, titulo, conteudo), ...] para os .md de fake_data/.

    O título é a primeira linha do arquivo (sem o # de markdown); se vazio, cai
    para o nome do arquivo. Usado pela indexação para gravar arquivo/titulo nos
    metadados de cada chunk.
    """
    docs: list[tuple[str, str, str]] = []
    if not FAKE_DATA_DIR.exists():
        return docs
    for path in sorted(FAKE_DATA_DIR.glob("*.md")):
        texto = path.read_text(encoding="utf-8").strip()
        primeira = texto.splitlines()[0] if texto else path.stem
        titulo = primeira.lstrip("#").strip() or path.stem
        docs.append((path.name, titulo, texto))
    return docs


def contar_chunks() -> int:
    """Conta os chunks atualmente indexados na collection.

    Lê direto as tabelas internas do langchain-postgres porque o PGVector não
    expõe uma contagem pública. Retorna 0 se as tabelas ainda não existem
    (base nunca inicializada).
    """
    sql = (
        "SELECT count(*) FROM langchain_pg_embedding e "
        "JOIN langchain_pg_collection c ON c.uuid = e.collection_id "
        "WHERE c.name = %s"
    )
    try:
        with psycopg.connect(db.get_dsn(), connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (COLLECTION_NAME,))
                return cur.fetchone()[0]
    except Exception:
        return 0


def listar_documentos() -> list[dict]:
    """Lista os documentos indexados, agregando os chunks por arquivo.

    Agrega a partir da metadata (arquivo/titulo) gravada em cada chunk. Lê as
    tabelas internas do langchain-postgres (não há API pública de listagem).
    Retorna [] se a base ainda não existe.
    """
    sql = (
        "SELECT e.cmetadata->>'arquivo' AS arquivo, "
        "       e.cmetadata->>'titulo' AS titulo, "
        "       count(*) AS chunks "
        "FROM langchain_pg_embedding e "
        "JOIN langchain_pg_collection c ON c.uuid = e.collection_id "
        "WHERE c.name = %s "
        "GROUP BY arquivo, titulo "
        "ORDER BY arquivo"
    )
    try:
        with psycopg.connect(db.get_dsn(), connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (COLLECTION_NAME,))
                linhas = cur.fetchall()
        return [
            {"arquivo": arquivo, "titulo": titulo, "chunks": chunks}
            for arquivo, titulo, chunks in linhas
        ]
    except Exception:
        return []


def remover_documento(arquivo: str) -> int:
    """Remove da collection todos os chunks de um arquivo. Retorna quantos.

    Idempotente: arquivo inexistente afeta 0 linhas (rowcount 0), sem erro.
    Exceções (ex.: base nunca inicializada) resolvem para 0 removidos, para a
    rota responder de forma limpa em vez de estourar 500.
    """
    sql = (
        "DELETE FROM langchain_pg_embedding e "
        "USING langchain_pg_collection c "
        "WHERE e.collection_id = c.uuid AND c.name = %s "
        "AND e.cmetadata->>'arquivo' = %s"
    )
    try:
        with psycopg.connect(db.get_dsn(), connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (COLLECTION_NAME, arquivo))
                removidos = cur.rowcount
            conn.commit()
        return removidos
    except Exception:
        return 0
