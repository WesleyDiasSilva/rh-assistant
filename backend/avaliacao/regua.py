"""Régua determinística da suíte de avaliação.

Funções puras que comparam o estado final do grafo com o gabarito declarado no
caso. Nenhuma delas chama modelo: tudo aqui é medido sobre dado que o próprio
sistema produziu (trajetória, fontes validadas, tool_calls, contadores), então o
resultado é reprodutível e não custa token.

O que NÃO é critério:

- `groundedness_score`: similaridade cosseno. Cega a negação, penaliza recusa
  correta e vale 0.0 na rota híbrida inteira (o node nem é visitado). É sempre
  reportado, nunca decide passou/falhou.
- `confianca`: autoavaliação do modelo, observada variando entre 0.6 e 1.0 para
  a mesma pergunta. Não mede qualidade.
- `categoria`: declarada como `str` livre no schema, não `Literal` — os valores
  sugeridos só existem na descrição do campo, nada impede uma variação.
"""
from __future__ import annotations

from typing import Any, NamedTuple

from app.tools import alerta_teto


class Veredito(NamedTuple):
    """Resultado da aplicação de um critério a um estado final.

    `avaliado=False` marca o critério que não pôde ser medido — por exemplo um
    juiz indisponível. Ele aparece na saída, mas não entra na conta de
    passou/falhou: um critério não medido não é um critério reprovado, e também
    não é um critério aprovado.
    """

    criterio: str
    ok: bool
    detalhe: str
    avaliado: bool = True


# --- Critérios ---------------------------------------------------------------

def _e_subsequencia(esperados: list[str], observados: list[str]) -> bool:
    """True se `esperados` aparece em `observados` na ordem, sem exigir adjacência."""
    restante = iter(observados)
    return all(no in restante for no in esperados)


def rota(estado: dict, esperado: list[str]) -> Veredito:
    """Os nós esperados foram percorridos, nessa ordem relativa.

    Comparação por SUBSEQUÊNCIA, nunca por igualdade de lista: a trajetória
    carrega todos os nós do turno, e a ordem em que os ramos de um fan-out
    paralelo escrevem nela é detalhe de implementação do LangGraph, não
    contrato do fluxo. Subsequência afirma o que importa — que o turno passou
    por estes nós, nesta ordem — sem amarrar a suíte ao resto.
    """
    observada = estado.get("trajetoria", [])
    if _e_subsequencia(esperado, observada):
        return Veredito("rota", True, " → ".join(esperado))
    return Veredito(
        "rota", False,
        f"esperado como subsequência {esperado}, observado {observada}",
    )


def fontes(estado: dict, esperado: list[str]) -> Veredito:
    """As políticas citadas são exatamente as esperadas.

    Comparação por CONJUNTO de nome de arquivo: a ordem de citação é escolha do
    modelo e não significa nada. O título não entra — `_validar_fontes` já o
    canoniza a partir dos metadados da base, então ele nunca divergiria.
    """
    observadas = {f.arquivo for f in estado["resposta"].fontes}
    esperadas = set(esperado)
    if observadas == esperadas:
        return Veredito("fontes", True, ", ".join(sorted(observadas)) or "(nenhuma)")
    faltando = esperadas - observadas
    sobrando = observadas - esperadas
    partes = []
    if faltando:
        partes.append(f"faltando {sorted(faltando)}")
    if sobrando:
        partes.append(f"sobrando {sorted(sobrando)}")
    return Veredito("fontes", False, "; ".join(partes))


def contem(estado: dict, esperado: list[str]) -> Veredito:
    """O texto da resposta contém os trechos que uma resposta correta precisa ter.

    Existe porque os demais critérios medem o CAMINHO — a rota percorrida, a
    política citada, a ferramenta escolhida — e não o CONTEÚDO. Sem este, uma
    resposta que cita a política de férias e afirma "15 dias" passa por todos os
    outros critérios: a fonte certa é um proxy de correção, não correção.

    Os trechos esperados vêm do gabarito da base (o número que a política de
    fato registra), não da resposta observada. Comparação sem diferenciar
    maiúsculas, porque a caixa do texto não é o que se afirma.
    """
    texto = estado["resposta"].resposta.lower()
    faltando = [t for t in esperado if t.lower() not in texto]
    if not faltando:
        return Veredito("contem", True, ", ".join(esperado))
    return Veredito("contem", False, f"não afirma {faltando}")


def recusa(estado: dict, esperado: bool) -> Veredito:
    """A resposta não se sustentou na base, e por isso não cita fonte alguma.

    Recusar quando a base não cobre o assunto é comportamento correto, não
    falha — e o sinal dessa recusa é a lista de fontes vazia, que
    `_validar_fontes` preserva. Critério separado de `fontes` porque expressa
    uma intenção diferente: aqui o gabarito é "não deve citar nada".
    """
    citadas = {f.arquivo for f in estado["resposta"].fontes}
    recusou = not citadas
    if recusou == esperado:
        return Veredito("recusa", True, "sem fontes" if recusou else "citou fontes")
    if esperado:
        return Veredito("recusa", False, f"esperava recusa, citou {sorted(citadas)}")
    return Veredito("recusa", False, "esperava fontes citadas, veio vazio")


def tool(estado: dict, esperado: dict) -> Veredito:
    """A ferramenta esperada foi escolhida, com os argumentos esperados.

    Lê `ai_msg.tool_calls`, que é a decisão do modelo antes da execução. O `id`
    da chamada nunca entra na comparação: é gerado por invocação.

    Não serve para a rota híbrida — lá `sub_dados` executa as tools inline e
    `ai_msg` fica None. Casos híbridos usam o critério `hibrido`.
    """
    ai_msg = estado.get("ai_msg")
    chamadas = getattr(ai_msg, "tool_calls", None) if ai_msg is not None else None
    if not chamadas:
        return Veredito("tool", False, "nenhuma tool foi chamada (ai_msg sem tool_calls)")
    observadas = [(c["name"], c["args"]) for c in chamadas]
    alvo = (esperado["name"], esperado.get("args", {}))
    if alvo in observadas:
        return Veredito("tool", True, f"{alvo[0]}({alvo[1]})")
    return Veredito(
        "tool", False,
        f"esperado {alvo[0]}({alvo[1]}), observado {observadas}",
    )


def tentativas(estado: dict, esperado: int) -> Veredito:
    """O ciclo de auto-correção rodou o número esperado de vezes."""
    observado = estado.get("tentativas", 0)
    if observado == esperado:
        return Veredito("tentativas", True, str(observado))
    return Veredito("tentativas", False, f"esperado {esperado}, observado {observado}")


def teto(estado: dict, esperado: str) -> Veredito:
    """O alerta de teto da política foi anexado ao resultado da ferramenta.

    O alerta é produzido em código por `alerta_teto` (não pelo modelo), então o
    critério compara com a saída dessa própria função: o que se afirma é que o
    grafo a acionou e costurou o resultado, quando `validar_teto` está ligado.
    """
    alerta = alerta_teto(esperado)
    if alerta is None:
        return Veredito(
            "teto", False,
            f"{esperado} não excede o teto no cadastro — caso não exercita o alerta",
        )
    conteudos = " ".join(tm.content for tm in estado.get("tool_messages", []))
    if alerta in conteudos:
        return Veredito("teto", True, f"alerta de teto presente para {esperado}")
    return Veredito("teto", False, f"alerta de teto ausente para {esperado}")


def hibrido(estado: dict, esperado: bool) -> Veredito:
    """Os dois ramos paralelos da rota híbrida produziram resultado.

    Mede a presença dos parciais `resposta_politica` e `resposta_dados`, que é o
    que o fan-out precisa entregar para `mesclar` ter o que unir. Não usa
    `ai_msg` (None nesta rota) nem compara o texto do ramo com literal algum.
    """
    pol = estado.get("resposta_politica")
    dad = estado.get("resposta_dados")
    ambos = pol is not None and dad is not None
    if ambos == esperado:
        detalhe = "resposta_politica e resposta_dados presentes" if ambos else "ramos ausentes"
        return Veredito("hibrido", True, detalhe)
    ausentes = [
        nome for nome, valor in (("resposta_politica", pol), ("resposta_dados", dad))
        if valor is None
    ]
    return Veredito("hibrido", False, f"ramos ausentes: {ausentes}")


# Critérios que a régua determinística sabe medir. Quem despacha os critérios de
# um caso confere esta tabela: um critério declarado e não reconhecido por
# ninguém é reportado como não avaliado, nunca contado como aprovado no silêncio.
CRITERIOS = {
    "rota": rota,
    "fontes": fontes,
    "contem": contem,
    "recusa": recusa,
    "tool": tool,
    "tentativas": tentativas,
    "teto": teto,
    "hibrido": hibrido,
}


def avaliar(estado: dict, criterios: dict[str, Any]) -> list[Veredito]:
    """Aplica ao estado final os critérios declarados no caso que a régua mede.

    Critérios que não estão em CRITERIOS são ignorados aqui — cabe a quem chama
    despachá-los (o juiz, por exemplo) ou reportá-los como não avaliados.
    """
    return [
        CRITERIOS[nome](estado, esperado)
        for nome, esperado in criterios.items()
        if nome in CRITERIOS
    ]
