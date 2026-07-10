import logging
from contextlib import asynccontextmanager

import psycopg
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row

from app import db, retrieval
from app.chat import ChatRequest, responder
from app.graph import compilar_grafo
from app.schemas import DocumentoBase, RemocaoBase, RespostaRH, Solicitacao
from app.tools import listar_solicitacoes

# Garante que os logs da aplicação (ex.: boot da extensão pgvector) apareçam;
# por padrão o uvicorn só configura os próprios loggers.
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:     %(name)s - %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Garante a extensão pgvector antes de atender requisições, independente do
    # estado do volume do Postgres (volume novo ou pré-existente sem a extensão).
    db.garantir_extensao_vector()

    # Checkpointer da memória de conversa: uma conexão psycopg v3 dedicada,
    # mantida viva por toda a aplicação. O PostgresSaver precisa de conexão
    # persistente — from_conn_string com `with` fecharia a conexão por request.
    # autocommit=True e row_factory=dict_row são requisitos do PostgresSaver.
    conn = psycopg.connect(
        db.get_dsn(),
        autocommit=True,
        prepare_threshold=0,
        row_factory=dict_row,
    )
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()  # cria/migra as tabelas checkpoint* (idempotente)
    # Grafo compilado com o checkpointer: só aqui, pois a compilação depende dele.
    app.state.grafo = compilar_grafo(checkpointer)
    try:
        yield
    finally:
        conn.close()


app = FastAPI(title="rh-assistant", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    db_ok = db.ping()
    return {"status": "ok", "db": "ok" if db_ok else "error"}


@app.post("/api/chat", response_model=RespostaRH)
def chat(req: ChatRequest, request: Request) -> RespostaRH:
    # Grafo compilado com o checkpointer é montado no lifespan e guardado em
    # app.state; injeta-se aqui para a thread da conversa manter memória.
    return responder(req, request.app.state.grafo)


@app.get("/api/solicitacoes", response_model=list[Solicitacao])
def solicitacoes() -> list[Solicitacao]:
    return listar_solicitacoes()


# --- Base de conhecimento (retrieval) ---------------------------------------

@app.get("/api/base", response_model=list[DocumentoBase])
def listar_base() -> list[DocumentoBase]:
    """Lista os documentos indexados, com a contagem de chunks por arquivo."""
    return [DocumentoBase(**doc) for doc in retrieval.listar_documentos()]


@app.post("/api/base/upload", response_model=DocumentoBase)
async def upload_base(file: UploadFile = File(...)) -> DocumentoBase:
    """Recebe um .md, deriva título/arquivo e indexa no mesmo gesto.

    Aceita apenas .md (outros formatos → 400, não 500). IDs estáveis por arquivo
    fazem upsert: reenviar o mesmo arquivo reindexa sem duplicar.
    """
    if not file.filename or not file.filename.lower().endswith(".md"):
        raise HTTPException(status_code=400, detail="Apenas arquivos .md são aceitos.")
    try:
        texto = (await file.read()).decode("utf-8").strip()
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Arquivo não é texto UTF-8 válido.")
    # Título = 1ª linha sem o # de markdown (mesma regra do seed); arquivo = nome.
    primeira = texto.splitlines()[0] if texto else file.filename
    titulo = primeira.lstrip("#").strip() or file.filename
    chunks = retrieval.indexar_documento(
        texto, {"arquivo": file.filename, "titulo": titulo}
    )
    return DocumentoBase(arquivo=file.filename, titulo=titulo, chunks=chunks)


@app.delete("/api/base/{arquivo}", response_model=RemocaoBase)
def deletar_base(arquivo: str) -> RemocaoBase:
    """Remove todos os chunks de um arquivo. Arquivo inexistente → 0 removidos."""
    removidos = retrieval.remover_documento(arquivo)
    return RemocaoBase(arquivo=arquivo, removidos=removidos)
