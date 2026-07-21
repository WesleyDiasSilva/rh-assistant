import logging
import os
from contextlib import asynccontextmanager

import psycopg
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg.rows import dict_row

from app import conversas, db, retrieval
from app.chat import ChatRequest, responder
from app.graph import compilar_grafo
from app.schemas import (
    ChatResponse,
    Conversa,
    DocumentoBase,
    MensagemHistorico,
    RemocaoBase,
    Solicitacao,
)
from app.tools import listar_solicitacoes

# Garante que os logs da aplicação (ex.: boot da extensão pgvector) apareçam;
# por padrão o uvicorn só configura os próprios loggers.
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:     %(name)s - %(message)s",
)

# --- LangFuse callback handler (instância global, lazy) ---------------------
# O LangSmith não requer código de instrumentação explícita: quando
# LANGCHAIN_TRACING_V2=true e LANGCHAIN_API_KEY estão definidas, o LangChain
# envia traces automaticamente. O LangFuse exige um callback registrado
# explicitamente nas invocações do grafo.

_langfuse_handler = None


def get_langfuse_handler():
    """Retorna o handler LangFuse, inicializando na primeira chamada.

    Retorna None se LANGFUSE_SECRET_KEY ou LANGFUSE_PUBLIC_KEY não estiverem
    configuradas, permitindo que o sistema funcione sem tracing ativo.
    """
    global _langfuse_handler
    if _langfuse_handler is None:
        secret = os.getenv("LANGFUSE_SECRET_KEY")
        public = os.getenv("LANGFUSE_PUBLIC_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        if secret and public:
            try:
                from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
                _langfuse_handler = LangfuseCallbackHandler()
                logging.getLogger(__name__).info("LangFuse callback handler inicializado.")
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "Não foi possível inicializar o LangFuse handler: %s", exc
                )
    return _langfuse_handler


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Garante a extensão pgvector antes de atender requisições, independente do
    # estado do volume do Postgres (volume novo ou pré-existente sem a extensão).
    db.garantir_extensao_vector()
    # Metadados de conversas (listagem/histórico): idempotente, cobre volume novo
    # e pré-existente, mesmo padrão da extensão pgvector acima.
    db.garantir_tabela_conversas()

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
    # O estado guarda um RespostaRH (Pydantic). Declarar esse tipo como módulo
    # permitido no serializer evita o aviso de "unregistered type" na
    # desserialização (get_state) — que versões futuras do LangGraph bloqueariam
    # — resolvendo na origem, sem silenciar log.
    serde = JsonPlusSerializer(allowed_msgpack_modules=[("app.schemas", "RespostaRH")])
    checkpointer = PostgresSaver(conn, serde=serde)
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


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request) -> ChatResponse:
    # Grafo compilado com o checkpointer é montado no lifespan e guardado em
    # app.state; injeta-se aqui para a thread da conversa manter memória.
    return responder(req, request.app.state.grafo)


@app.get("/api/solicitacoes", response_model=list[Solicitacao])
def solicitacoes() -> list[Solicitacao]:
    return listar_solicitacoes()


# --- Conversas (metadados e histórico) --------------------------------------

@app.get("/api/conversas", response_model=list[Conversa])
def listar_conversas() -> list[Conversa]:
    """Lista as conversas registradas, da mais recente para a mais antiga."""
    return [Conversa(**c) for c in conversas.listar()]


@app.get("/api/conversas/{conversa_id}", response_model=list[MensagemHistorico])
def historico_conversa(conversa_id: str, request: Request) -> list[MensagemHistorico]:
    """Devolve o histórico de uma conversa, para hidratar o chat na interface.

    A fonte é o checkpointer (estado da thread): get_state expõe o campo
    `mensagens` acumulado. Só os papéis usuário/assistente entram, na ordem.
    Conversa não registrada → 404.
    """
    if not conversas.existe(conversa_id):
        raise HTTPException(status_code=404, detail="Conversa não encontrada.")
    snap = request.app.state.grafo.get_state(
        {"configurable": {"thread_id": conversa_id}}
    )
    papel = {"human": "usuario", "ai": "assistente"}
    return [
        MensagemHistorico(papel=papel[m.type], texto=m.content)
        for m in snap.values.get("mensagens", [])
        if getattr(m, "type", None) in papel
    ]


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
