# rh-assistant — branch `demonstracao`

> **Esta branch é apenas demonstração**. As respostas vêm de matching por palavra-chave sobre 6 documentos fake em `backend/fake_data/`. Não há LLM, nem chamada externa, nem API key — é tudo offline e simulado. Serve para a turma ver o destino do produto antes de começar a Aula 1.

Assistente de RH construído ao longo do curso de LangChain.

Stack: **React + Vite + Tailwind** (frontend) · **FastAPI** (backend) · **Postgres 16** (db). Tudo orquestrado por `docker compose`.

## Branches

- **`aula01-inicio`** — esqueleto. Tudo sobe no Docker, mas `/api/chat` retorna apenas um mock. Ponto de partida da Aula 1.
- **`demonstracao`** *(esta branch)* — produto fake completo, com matching por palavra-chave sobre 6 documentos de RH simulados.

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
