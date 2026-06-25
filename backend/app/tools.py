"""
Tools de RH para o assistente de férias.

Cada função decorada com @tool vira uma ferramenta que o modelo pode chamar.
A docstring é o que o modelo lê para decidir quando e como usar a tool, por
isso ela descreve claramente o que a tool faz e quais argumentos espera.

- consultar_saldo_ferias: leitura do saldo de férias de um funcionário.
- registrar_solicitacao_ferias: registra uma solicitação de férias.

Os dados ficam em arquivos simples: o saldo em fake_data/saldos.json e as
solicitações em solicitacoes.jsonl (uma por linha).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool


BACKEND_DIR = Path(__file__).resolve().parent.parent
SALDOS_PATH = BACKEND_DIR / "fake_data" / "saldos.json"
SOLICITACOES_PATH = BACKEND_DIR / "solicitacoes.jsonl"


@tool
def consultar_saldo_ferias(funcionario: str) -> str:
    """Consulta o saldo de férias de um funcionário pelo nome.

    Use esta ferramenta quando o usuário quiser saber quantos dias de férias
    alguém tem disponíveis ou já usou. Recebe o nome do funcionário (ex.: "Ana")
    e devolve os dias disponíveis e os dias já usados. Se o funcionário não for
    encontrado, devolve "Funcionário não encontrado".
    """
    saldos = json.loads(SALDOS_PATH.read_text(encoding="utf-8"))
    dados = saldos.get(funcionario)
    if dados is None:
        return "Funcionário não encontrado"
    return (
        f"{funcionario} tem {dados['dias_disponiveis']} dias de férias "
        f"disponíveis e {dados['dias_usados']} dias já usados."
    )


@tool
def registrar_solicitacao_ferias(funcionario: str, dias: int, periodo: str) -> str:
    """Registra uma solicitação de férias de um funcionário.

    Use esta ferramenta quando o usuário quiser solicitar/agendar férias.
    Recebe o nome do funcionário, a quantidade de dias (inteiro) e o período
    desejado (ex.: "julho/2026" ou "01/07 a 15/07"). A solicitação é gravada
    e a função devolve uma confirmação com os dados registrados.
    """
    registro = {
        "funcionario": funcionario,
        "dias": dias,
        "periodo": periodo,
        "timestamp": datetime.now().isoformat(),
    }
    with SOLICITACOES_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")
    return (
        f"Solicitação registrada: {dias} dias em {periodo} para {funcionario}"
    )
