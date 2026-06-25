"""
Schemas Pydantic da resposta estruturada do assistente de RH.

`RespostaRH` é tanto o formato de saída da API quanto o esquema usado em
`model.with_structured_output(...)`: o modelo é obrigado a preencher esses
campos, evitando a resposta em string solta.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Fonte(BaseModel):
    """Política de RH utilizada para compor a resposta."""

    arquivo: str = Field(description="Nome do arquivo .md da política.")
    titulo: str = Field(description="Título da política.")


class RespostaRH(BaseModel):
    """Resposta estruturada a uma pergunta sobre políticas de RH."""

    resposta: str = Field(description="Resposta clara e objetiva à pergunta.")
    fontes: list[Fonte] = Field(
        default_factory=list,
        description="Apenas as políticas realmente usadas para responder.",
    )
    categoria: str = Field(
        description=(
            "Categoria da pergunta: ferias, home-office, beneficios, "
            "reembolso, horario, licencas ou outro."
        )
    )
    confianca: float = Field(
        ge=0.0,
        le=1.0,
        description="Confiança na resposta, de 0 a 1.",
    )
