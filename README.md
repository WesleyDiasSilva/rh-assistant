# rh-assistant

Assistente de RH construído ao longo do curso de LangChain.

Stack: **React + Vite + Tailwind** (frontend) · **FastAPI** (backend) · **Postgres 16** (db). Tudo orquestrado por `docker compose`.

## Branches

- **`aula01-inicio`** — esqueleto. Tudo sobe no Docker, mas `/api/chat` retorna apenas um mock. Ponto de partida da Aula 1.
- **`demonstracao`** — produto fake completo, com matching por palavra-chave sobre 6 documentos de RH simulados. Serve para a turma ver onde queremos chegar. **Não usa LLM nem API key — é tudo offline.**

## Pré-requisito

Apenas **Docker** com Docker Compose v2.

## Como rodar

```bash
cp .env.example .env
docker compose up
```

Pronto. Abra:

- **UI** → http://localhost:5173
- **Saúde do backend** → http://localhost:8000/health

Para parar: `Ctrl+C` e depois `docker compose down`.
