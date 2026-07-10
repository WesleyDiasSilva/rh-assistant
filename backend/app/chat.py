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

from app import conversas
from app.schemas import RespostaRH


# --- Contrato de entrada da API ---------------------------------------------

class ChatRequest(BaseModel):
    pergunta: str
    # Identifica a conversa (thread) no checkpointer: mesmo id → mesma memória.
    # O front gera um uuid por conversa e o reenvia em cada pergunta.
    conversa_id: str
    # Quando ligado, o saldo consultado é validado contra o teto da política
    # (a tool sinaliza inconsistências). Controlado pela UI.
    validar_teto: bool = False
    # Quando ligado, se a busca não sustentar a resposta (nenhuma fonte), o
    # sistema reescreve a pergunta (removendo ruído) e tenta uma única vez mais.
    # Auto-correção reativa, não pré-processamento. Controlado pela UI.
    auto_corrigir: bool = False


# --- Rota -------------------------------------------------------------------

def responder(req: ChatRequest, grafo) -> RespostaRH:
    """Invoca o grafo (compilado com checkpointer) para a thread da conversa.

    O grafo compilado é injetado (montado no lifespan com o PostgresSaver), não
    importado como singleton de módulo — a compilação depende do checkpointer.
    """
    try:
        estado_final = grafo.invoke(
            {
                "pergunta": req.pergunta,
                "auto_corrigir": req.auto_corrigir,
                "validar_teto": req.validar_teto,
                # Reset explícito dos intermediários: o checkpointer persiste o
                # estado da thread, então sem zerar estes campos valores do turno
                # anterior vazariam para o atual. `mensagens` fica de fora de
                # propósito — é o único campo que deve acumular (add_messages).
                "categoria_triagem": "",
                "consulta": "",
                "tentativas": 0,
                "ai_msg": None,
                "chunks": [],
                "tool_messages": [],
            },
            config={"configurable": {"thread_id": req.conversa_id}},
        )
        # Registra o metadado da conversa no primeiro turno (no-op nos demais).
        # Fica na camada de API: título é metadado de produto, não estado do grafo.
        conversas.registrar_se_nova(req.conversa_id, req.pergunta)
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
