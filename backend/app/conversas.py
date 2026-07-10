"""Metadados de conversas para a listagem e o histórico do produto.

Guarda apenas os metadados de cada conversa — id (o mesmo thread_id do
checkpointer), título e data de criação — numa tabela própria (`conversas`). O
estado da conversa em si (as mensagens) continua no checkpointer; aqui ficam só
os dados que a interface usa para listar e rotular conversas.
"""
from __future__ import annotations

import logging

import psycopg

from app import db

logger = logging.getLogger(__name__)

# Tamanho máximo do título derivado da primeira pergunta. Acima disso, corta e
# acrescenta reticências — o título é um rótulo curto para a lista, não o texto.
TITULO_MAX = 60


def _titulo_de(pergunta: str) -> str:
    """Deriva o título da conversa a partir da primeira pergunta (truncado).

    Normaliza espaços e quebras de linha e, se exceder TITULO_MAX, corta e
    acrescenta reticências.
    """
    texto = " ".join(pergunta.split())
    if len(texto) > TITULO_MAX:
        return texto[:TITULO_MAX].rstrip() + "…"
    return texto


def registrar_se_nova(conversa_id: str, primeira_pergunta: str) -> None:
    """Registra a conversa na primeira vez que ela aparece.

    Idempotente e à prova de corrida via ON CONFLICT DO NOTHING: só o primeiro
    turno insere a linha (com o título derivado da pergunta); turnos seguintes
    não têm efeito. Falhas são logadas e não interrompem a resposta ao usuário —
    o metadado é auxiliar.
    """
    sql = (
        "INSERT INTO conversas (id, titulo) VALUES (%s, %s) "
        "ON CONFLICT (id) DO NOTHING"
    )
    try:
        with psycopg.connect(db.get_dsn(), connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (conversa_id, _titulo_de(primeira_pergunta)))
            conn.commit()
    except Exception as exc:
        logger.warning("Não foi possível registrar a conversa %s: %s", conversa_id, exc)


def existe(conversa_id: str) -> bool:
    """Indica se a conversa já está registrada na tabela de metadados."""
    try:
        with psycopg.connect(db.get_dsn(), connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM conversas WHERE id = %s", (conversa_id,))
                return cur.fetchone() is not None
    except Exception:
        return False


def listar() -> list[dict]:
    """Lista as conversas registradas, da mais recente para a mais antiga."""
    sql = "SELECT id, titulo, criada_em FROM conversas ORDER BY criada_em DESC"
    try:
        with psycopg.connect(db.get_dsn(), connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                linhas = cur.fetchall()
        return [
            {"id": id_, "titulo": titulo, "criada_em": criada_em}
            for id_, titulo, criada_em in linhas
        ]
    except Exception:
        return []
