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
from app.schemas import ChatResponse


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

def responder(req: ChatRequest, grafo) -> ChatResponse:
    """Invoca o grafo (compilado com checkpointer) para a thread da conversa.

    O grafo compilado é injetado (montado no lifespan com o PostgresSaver), não
    importado como singleton de módulo — a compilação depende do checkpointer.
    """
    from app.main import get_langfuse_handler, get_langfuse_client
    try:
        config: dict = {"configurable": {"thread_id": req.conversa_id}}
        handler = get_langfuse_handler()
        if handler:
            config["callbacks"] = [handler]

        estado_final = grafo.invoke(
            {
                "pergunta": req.pergunta,
                "auto_corrigir": req.auto_corrigir,
                "validar_teto": req.validar_teto,
                # Reset explícito dos intermediários: o checkpointer persiste o
                # estado da thread, então sem zerar estes campos valores do turno
                # anterior vazariam para o atual. `mensagens` fica de fora de
                # propósito — é o único campo que deve acumular (add_messages).
                # `trajetoria` também é zerada: é rastro do turno, não da conversa.
                "categoria_triagem": "",
                "tipo_consulta": "",
                "consulta": "",
                "tentativas": 0,
                "ai_msg": None,
                "chunks": [],
                "tool_messages": [],
                "pergunta_dados": "",
                "pergunta_politica": "",
                "resposta_politica": None,
                "resposta_dados": None,
                "trajetoria": None,
                "groundedness_score": 0.0,
            },
            config=config,
        )
        # Envia o groundedness score para o LangFuse como métrica nomeada.
        # handler.last_trace_id contém o UUID que o LangFuse gerou para este
        # invoke — é o identificador correto para associar o score à trace certa.
        score = estado_final.get("groundedness_score", 0.0)
        if score > 0 and handler:
            trace_id = getattr(handler, "last_trace_id", None)
            if trace_id:
                client = get_langfuse_client()
                if client:
                    try:
                        client.create_score(
                            trace_id=trace_id, name="groundedness", value=score
                        )
                    except Exception as exc:
                        import logging as _log
                        _log.getLogger(__name__).warning(
                            "[groundedness] falha ao logar score no LangFuse: %s", exc
                        )

        # Registra o metadado da conversa no primeiro turno (no-op nos demais).
        # Fica na camada de API: título é metadado de produto, não estado do grafo.
        conversas.registrar_se_nova(req.conversa_id, req.pergunta)
        r = estado_final["resposta"]
        return ChatResponse(
            resposta=r.resposta,
            fontes=r.fontes,
            categoria=r.categoria,
            confianca=r.confianca,
            trajetoria=estado_final.get("trajetoria", []),
            groundedness_score=estado_final.get("groundedness_score", 0.0),
        )
    except Exception as exc:  # ex.: sem ANTHROPIC_API_KEY, falha de rede/API
        return ChatResponse(
            resposta=(
                "Não consegui consultar o assistente agora. Verifique se a "
                "ANTHROPIC_API_KEY está configurada e tente novamente. "
                f"(detalhe: {exc})"
            ),
            fontes=[],
            categoria="outro",
            confianca=0.0,
            trajetoria=[],
        )
