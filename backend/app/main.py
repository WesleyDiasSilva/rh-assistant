from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import db
from app.chat import ChatRequest, ChatResponse, responder

app = FastAPI(title="rh-assistant", version="0.1.0")

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
def chat(req: ChatRequest) -> ChatResponse:
    return responder(req)
