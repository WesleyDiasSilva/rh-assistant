import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import db
from app.chat import ChatRequest, responder
from app.schemas import RespostaRH, Solicitacao
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
    yield


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
def chat(req: ChatRequest) -> RespostaRH:
    return responder(req)


@app.get("/api/solicitacoes", response_model=list[Solicitacao])
def solicitacoes() -> list[Solicitacao]:
    return listar_solicitacoes()
