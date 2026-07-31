"""Envio dos resultados da suíte para a plataforma de observabilidade (LangFuse).

Cada caso já produz uma trace, porque o handler de callback é anexado ao invoke
do grafo. Este módulo acrescenta a essa trace os vereditos da suíte como scores
nomeados — "regua" e "juiz", valor 1 ou 0 — para o resultado da avaliação ficar
ao lado da execução que o produziu, e não só no terminal de quem rodou.

Best-effort em todos os pontos: sem as chaves no ambiente tudo vira no-op e a
suíte roda igual; falha de envio é reportada e não altera veredito nenhum. A
plataforma é destino do resultado, nunca parte do critério.

Sequencial por construção: o handler guarda o id da última trace num atributo de
instância (`last_trace_id`), então ler esse id só é confiável imediatamente após
o invoke correspondente. Rodar casos em paralelo embaralharia a associação entre
score e trace.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime

# Identificador desta rodada, usado como sessão no LangFuse: agrupa as traces de
# todos os casos de uma execução da suíte numa única vista.
RODADA_ID = f"avaliacao-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"

_handler = None
_client = None
_tentou_inicializar = False
_falhas: list[str] = []


def _inicializar() -> None:
    """Cria handler e client na primeira necessidade, se houver chaves no ambiente."""
    global _handler, _client, _tentou_inicializar
    if _tentou_inicializar:
        return
    _tentou_inicializar = True
    if not (os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY")):
        return
    try:
        from langfuse import get_client
        from langfuse.langchain import CallbackHandler

        # No SDK 4.x o client é um singleton configurado por ambiente, e o
        # handler de callback o reaproveita — não recebe chaves no construtor.
        _client = get_client()
        _handler = CallbackHandler()
    except Exception as exc:
        _falhas.append(f"inicialização: {type(exc).__name__}: {exc}")


def disponivel() -> bool:
    """True quando há handler ativo, isto é, quando as traces estão sendo enviadas."""
    _inicializar()
    return _handler is not None


def callbacks() -> list:
    """Callbacks a anexar ao config do invoke. Vazio quando não há plataforma."""
    _inicializar()
    return [_handler] if _handler is not None else []


def atributos_trace(caso_id: str) -> dict:
    """Metadados que dão nome e agrupamento à trace do caso.

    O nome da trace passa a ser o id do caso, e a sessão agrupa a rodada: é o
    que permite achar um caso específico na interface sem procurar por horário.
    """
    if not disponivel():
        return {}
    return {
        "langfuse_trace_name": caso_id,
        "langfuse_session_id": RODADA_ID,
        "langfuse_tags": ["avaliacao"],
    }


def trace_id_recente() -> str | None:
    """Id da trace do invoke mais recente. Só é confiável logo após o invoke."""
    return getattr(_handler, "last_trace_id", None) if _handler is not None else None


def _enviar_score(trace_id: str, name: str, valor: int, comment: str, caso_id: str) -> None:
    """Envia um score. Assinatura keyword-only, conforme o SDK 4.x."""
    try:
        _client.create_score(
            name=name,
            value=float(valor),
            trace_id=trace_id,
            data_type="NUMERIC",
            comment=comment[:1000] or None,
            metadata={"caso": caso_id, "rodada": RODADA_ID},
        )
    except Exception as exc:
        _falhas.append(f"score {name} de {caso_id}: {type(exc).__name__}: {exc}")


def enviar_scores(trace_id: str | None, caso_id: str, vereditos: list) -> None:
    """Envia um score por família de critério: "regua" e "juiz", valor 1 ou 0.

    O critério do juiz vira o score "juiz" e os demais, agregados, o score
    "regua" — 1 só quando todos os critérios determinísticos medidos passaram.
    Critério não avaliado fica fora: não há veredito para registrar.
    """
    if _client is None or not trace_id:
        return
    juiz = [v for v in vereditos if v.criterio == "juiz" and v.avaliado]
    regua = [v for v in vereditos if v.criterio != "juiz" and v.avaliado]

    if regua:
        reprovados = [v.criterio for v in regua if not v.ok]
        _enviar_score(
            trace_id,
            "regua",
            0 if reprovados else 1,
            f"reprovou em: {', '.join(reprovados)}" if reprovados else "todos os critérios determinísticos aprovados",
            caso_id,
        )
    for v in juiz:
        # O motivo do juiz vai como comentário do score: é a justificativa que
        # torna o 0 ou 1 legível na interface.
        _enviar_score(trace_id, "juiz", 1 if v.ok else 0, v.detalhe, caso_id)


def finalizar() -> list[str]:
    """Descarrega o buffer e devolve as falhas acumuladas, para o runner reportar.

    O flush é obrigatório num processo curto: o SDK envia em lote e o processo
    terminaria antes do envio, fazendo os scores simplesmente não aparecerem.
    """
    if _client is not None:
        try:
            _client.flush()
        except Exception as exc:
            _falhas.append(f"flush: {type(exc).__name__}: {exc}")
    return _falhas
