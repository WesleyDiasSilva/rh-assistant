"""
Aula 1 — ponto de partida.

A rota /api/chat ainda não está implementada. Retorna um mock fixo.
Ao longo do curso, esta função será substituída por uma cadeia LangChain
real, com recuperação de documentos e LLM.
"""
from pydantic import BaseModel


class ChatRequest(BaseModel):
    pergunta: str


class Fonte(BaseModel):
    arquivo: str
    titulo: str


class ChatResponse(BaseModel):
    resposta: str
    fontes: list[Fonte]


def responder(req: ChatRequest) -> ChatResponse:
    return ChatResponse(
        resposta="Em breve. Esta rota será implementada ao longo do curso.",
        fontes=[],
    )
