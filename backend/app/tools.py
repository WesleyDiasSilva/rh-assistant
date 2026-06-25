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

# Máximo de dias de férias previsto na política. A regra de negócio mora aqui,
# no código do sistema — não depende de o modelo "lembrar" da política.
TETO_FERIAS = 30


def _dados_saldo(funcionario: str) -> dict | None:
    """Lê saldos.json e devolve os dados do funcionário, ou None se não existir."""
    saldos = json.loads(SALDOS_PATH.read_text(encoding="utf-8"))
    return saldos.get(funcionario)


def alerta_teto(funcionario: str) -> str | None:
    """Alerta se o saldo cadastrado do funcionário exceder o teto da política.

    Retorna None se estiver dentro do teto ou se o funcionário não existir.
    """
    dados = _dados_saldo(funcionario)
    if dados is None or dados["dias_disponiveis"] <= TETO_FERIAS:
        return None
    return (
        f"Atenção: o cadastro indica {dados['dias_disponiveis']} dias para "
        f"{funcionario}, mas o máximo previsto na política é {TETO_FERIAS} "
        "dias. Trata-se de uma inconsistência cadastral — confirme com o RH."
    )


def listar_solicitacoes() -> list[dict]:
    """Lê solicitacoes.jsonl e devolve as solicitações, da mais recente para a
    mais antiga. Se o arquivo não existir ou estiver vazio, devolve [].

    Não é uma tool: é usada pela rota GET /api/solicitacoes para exibir o que a
    tool de registro gravou.
    """
    if not SOLICITACOES_PATH.exists():
        return []
    registros: list[dict] = []
    for linha in SOLICITACOES_PATH.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if linha:
            registros.append(json.loads(linha))
    registros.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return registros


@tool
def consultar_saldo_ferias(funcionario: str) -> str:
    """Consulta o saldo de férias de um funcionário pelo nome.

    Use esta ferramenta quando o usuário quiser saber quantos dias de férias
    alguém tem disponíveis ou já usou. Recebe o nome do funcionário (ex.: "Ana")
    e devolve os dias disponíveis e os dias já usados. Se o funcionário não for
    encontrado, devolve "Funcionário não encontrado".
    """
    dados = _dados_saldo(funcionario)
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
