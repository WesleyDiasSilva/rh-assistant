"""
Ponto de entrada da rota /api/chat.

O fluxo de decisão em si (decidir entre tool e política, retrieval, execução de
tools e formatação da resposta) mora em app/graph.py, modelado como um
StateGraph. Aqui ficam só o contrato de entrada da API (ChatRequest) e a função
responder(), que monta o estado inicial a partir do request, invoca o grafo
compilado e devolve o RespostaRH do estado final.
"""
from __future__ import annotations

from pydantic import BaseModel

from app.graph import grafo
from app.schemas import RespostaRH


# --- Contrato de entrada da API ---------------------------------------------

class ChatRequest(BaseModel):
    pergunta: str
    # Quando ligado, o saldo consultado é validado contra o teto da política
    # (a tool sinaliza inconsistências). Controlado pela UI.
    validar_teto: bool = False
    # Quando ligado, a pergunta é normalizada (artigos/palavras supérfluas
    # removidos) antes de ser usada. A query normalizada alimenta tanto a busca
    # (estabiliza o ranking do top-k) quanto a geração. Controlado pela UI.
    reescrever_pergunta: bool = False


# --- Rota -------------------------------------------------------------------

def responder(req: ChatRequest) -> RespostaRH:
    try:
        estado_final = grafo.invoke(
            {
                "pergunta": req.pergunta,
                "reescrever_pergunta": req.reescrever_pergunta,
                "validar_teto": req.validar_teto,
            }
        )
        return estado_final["resposta"]
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
