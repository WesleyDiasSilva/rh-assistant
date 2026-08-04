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
- **`aula04-fim`** — refinamento do retrieval: a busca passa a expor o score de
  similaridade nos logs e há um toggle opcional "Reescrever pergunta" que
  normaliza a pergunta (remove ruído) antes de alimentar busca e geração.
- **`aula05-inicio` / `aula05-fim`** — orquestração do chat migrada para um grafo
  de estados (LangGraph), com triagem de escopo na entrada e um ciclo de
  auto-correção que reescreve a consulta quando a busca não traz bons resultados.
- **`aula06-inicio` / `aula06-fim`** — memória de conversa por thread com
  checkpointer Postgres: a busca é contextualizada pelo histórico, e a UI ganha
  navegação lateral com listagem e histórico de conversas.
- **`aula07-inicio` / `aula07-fim`** — rota conversacional que responde a partir
  do histórico, decomposição de perguntas híbridas (consulta de dados e busca de
  política em paralelo) e exibição da trajetória de nós percorrida em cada resposta.
- **`aula08-inicio` / `aula08-fim`** — observabilidade: instrumentação LangSmith e
  LangFuse, node de avaliação de groundedness e exibição do score no frontend,
  com o score vinculado à trace correta na plataforma.
- **`aula09-inicio` / `aula09-fim`** — suíte de avaliação automatizada em
  `backend/avaliacao/`: casos declarados em `casos.json`, régua de critérios
  determinísticos, critério julgado por modelo, pré-condição do estado da base,
  modo rápido, comparação com a rodada anterior e envio dos vereditos como
  scores no LangFuse.
- **`aula10-inicio` / `aula10-fim`** — o modelo de embedding passa a ser
  registrado na metadata de cada chunk, e o boot avisa quando o modelo
  configurado diverge do que gerou os vetores da base.
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

Cada chunk registra em sua metadata o modelo de embedding que gerou o seu vetor.
Trocar esse modelo não é mudança de configuração: cada modelo projeta o texto num
espaço próprio, e distância entre vetores de espaços diferentes não mede
semelhança. Quando as dimensões divergem, a operação falha de forma explícita;
quando coincidem, a busca continua rodando e passa a devolver os documentos
errados sem sinal nenhum. Por isso o boot confere o modelo registrado nos chunks
contra o configurado e **avisa no log** quando divergem — não bloqueia, porque
reindexar é decisão de quem opera. Para resolver, reconstrua a base com
`limpar_base.py` + `seed_politicas.py`.

Quantos chunks a busca retorna é controlado por `TOP_K` (default 4). Serve para
medir o efeito do parâmetro sem editar código:

```bash
docker compose exec -e TOP_K=1 backend python -m avaliacao.rodar --rapido
```

## Suíte de avaliação

`backend/avaliacao/` mede o comportamento do grafo contra casos declarados em
`casos.json`. Não usa framework de teste: um caso é um dicionário de dados e a
suíte é um laço sobre eles, invocando o grafo compilado sem checkpointer.

```bash
# Suíte completa (~1min20s).
docker compose exec backend python -m avaliacao.rodar

# Conjunto de demonstração, 5 casos (~30s).
docker compose exec backend python -m avaliacao.rodar --rapido

# Um caso, repetido, para expor oscilação.
docker compose exec backend python -m avaliacao.rodar --caso tool-saldo-ana --repeticoes 5
```

Cada caso declara seus critérios. Os determinísticos são medidos por funções
puras, sem chamar modelo (`regua.py`): a rota percorrida, as políticas citadas,
o conteúdo que a resposta precisa afirmar, a ferramenta escolhida, os
contadores, o alerta de teto. O que só se julga lendo o texto vai para um
critério julgado por modelo (`juiz.py`), com veredito binário e justificativa.

O score de groundedness é sempre reportado e nunca decide passou/falhou: é uma
similaridade cosseno, cega a negação e zerada fora da rota de política.

Dois casos declaram `esperado_vermelho`: reprovam de propósito, documentando
comportamentos conhecidos do sistema. O código de saída sinaliza *mudança* em
relação ao declarado — caso verde que reprovou, ou vermelho esperado que passou.

Antes de medir, o runner confere se a base vetorial está no estado que os casos
pressupõem e recusa rodar se não estiver, porque com um documento a menos o
vermelho fica ambíguo entre defeito do sistema e base suja. Use `--ignorar-base`
para medir mesmo assim.

Ao fim de cada rodada o resultado é comparado com o da rodada anterior
(`delta: 12/14 → 9/14, 3 regressões`). A base de comparação fica em
`avaliacao/.ultima-rodada.json`, fora do git.

Com as chaves do LangFuse no ambiente, cada caso gera uma trace nomeada com o id
do caso e recebe dois scores, `regua` e `juiz`, com a justificativa do juiz no
comentário. Sem as chaves, a suíte roda igual.
