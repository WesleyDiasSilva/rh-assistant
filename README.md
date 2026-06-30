# rh-assistant

Assistente de RH que responde perguntas sobre políticas internas com base nos
documentos da empresa, usando RAG (busca por similaridade) e citando a política
de origem de cada resposta.

Stack: **React + Vite + Tailwind** (frontend) · **FastAPI** (backend) ·
**Postgres 16 + pgvector** (base vetorial). Tudo orquestrado por `docker compose`.

Geração de resposta pela Anthropic (Claude); embeddings para o retrieval pela
OpenAI.

## Branches

Convenção: `aulaXX-inicio` é o ponto de partida e `aulaXX-fim` o checkpoint
correspondente.

- **`aula01-inicio`** — esqueleto. Tudo sobe no Docker, mas `/api/chat` retorna
  apenas um mock.
- **`aula01-fim`** — primeira chain com LangChain + Claude.
- **`aula02-inicio` / `aula02-fim`** — saída estruturada (`RespostaRH`), tool
  calling (consulta e registro de férias) e roteamento de intenção.
- **`aula03-inicio` / `aula03-fim`** — RAG com pgvector: indexação das políticas,
  retrieval no fluxo informativo e gestão da base (listar, enviar e remover
  documentos) via API e UI.
- **`demonstracao`** — produto de referência completo, com matching por
  palavra-chave sobre os documentos de RH simulados. Não usa LLM nem API key —
  é tudo offline.

## Pré-requisito

**Docker** com Docker Compose v2.

## Como rodar

```bash
cp .env.example .env
# preencha ANTHROPIC_API_KEY (geração) e OPENAI_API_KEY (embeddings) no .env
docker compose up
```

Abra:

- **UI** → http://localhost:5173
- **Saúde do backend** → http://localhost:8000/health

Para parar: `Ctrl+C` e depois `docker compose down`.

## Base de conhecimento vetorial

As políticas em `backend/fake_data/*.md` são indexadas em uma base vetorial
(pgvector) para busca por similaridade. A base pode ser gerenciada pela UI
(painel "Base de conhecimento": listar, enviar e remover documentos) ou pelos
scripts abaixo, que rodam dentro do container do backend:

```bash
# Esvazia a base (remove todos os chunks). Idempotente.
docker compose exec backend python limpar_base.py

# Indexa as políticas. Idempotente: rodar de novo não duplica os chunks.
docker compose exec backend python seed_politicas.py
```

Requer `OPENAI_API_KEY` no `.env` (os embeddings são gerados pela OpenAI).
