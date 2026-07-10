import logging
import os

import psycopg

logger = logging.getLogger(__name__)


def get_dsn() -> str:
    return (
        f"host={os.getenv('DB_HOST', 'db')} "
        f"port={os.getenv('DB_PORT', '5432')} "
        f"user={os.getenv('DB_USER', 'postgres')} "
        f"password={os.getenv('DB_PASSWORD', 'postgres')} "
        f"dbname={os.getenv('DB_NAME', 'rh_assistant')}"
    )


def get_sqlalchemy_url() -> str:
    """URL no formato esperado pelo langchain-postgres (PGVector), via psycopg3.

    Reaproveita as mesmas variáveis de ambiente do get_dsn(), mas no formato
    SQLAlchemy: postgresql+psycopg://user:senha@host:porta/db
    """
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "postgres")
    host = os.getenv("DB_HOST", "db")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "rh_assistant")
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{name}"


def garantir_extensao_vector() -> bool:
    """Garante que a extensão pgvector exista no banco.

    Idempotente: roda CREATE EXTENSION IF NOT EXISTS vector. Cobre o caso de
    volumes pré-existentes onde o script de init não rodou. Falhas transitórias
    são logadas e não derrubam o boot da aplicação.
    """
    try:
        with psycopg.connect(get_dsn(), connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
        logger.info("Extensão pgvector garantida (CREATE EXTENSION IF NOT EXISTS vector).")
        return True
    except Exception as exc:
        logger.warning("Não foi possível garantir a extensão pgvector no boot: %s", exc)
        return False


def garantir_tabela_conversas() -> bool:
    """Garante a tabela de metadados de conversas.

    Idempotente (CREATE TABLE IF NOT EXISTS), no mesmo espírito de
    garantir_extensao_vector: cobre volume novo e pré-existente. Guarda só os
    metadados de produto (título e data); o estado da conversa em si vive no
    checkpointer. Falhas transitórias são logadas e não derrubam o boot.
    """
    sql = (
        "CREATE TABLE IF NOT EXISTS conversas ("
        "id TEXT PRIMARY KEY, "
        "titulo TEXT NOT NULL, "
        "criada_em TIMESTAMPTZ NOT NULL DEFAULT now()"
        ")"
    )
    try:
        with psycopg.connect(get_dsn(), connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
        logger.info("Tabela de conversas garantida (CREATE TABLE IF NOT EXISTS conversas).")
        return True
    except Exception as exc:
        logger.warning("Não foi possível garantir a tabela de conversas no boot: %s", exc)
        return False


def ping() -> bool:
    try:
        with psycopg.connect(get_dsn(), connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True
    except Exception:
        return False
