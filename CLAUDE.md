# CLAUDE.md — rh-assistant

## Contexto
Assistente de RH que responde perguntas sobre políticas internas com base em
documentos. Stack: React + Vite (frontend), FastAPI (backend), Postgres,
docker compose.

## Estrutura de branches
- `demonstracao` — versão de referência do produto, com a interface e as
  funcionalidades completas.
- `aulaXX-inicio` e `aulaXX-fim` — pontos de partida e checkpoints usados ao
  longo do curso.

## README por branch
- Cada branch mantém um README atualizado que descreve o que o projeto é, o
  estado atual daquela branch e as branches existentes no repositório.
- Mantenha o README coerente com o estado real da branch. Não liste branches
  inexistentes.

## Convenções de código e UI
- A interface é a de um produto de assistente de RH. Mantenha textos de UI,
  comentários e docstrings neutros e descritivos — sem referências a curso,
  aula ou a estados temporários de implementação.
- Prefira soluções simples. Ex.: a base de conhecimento são arquivos .md numa
  pasta, lidos pelo backend — sem banco, storage ou upload.

## Commits e segurança
- NUNCA commite o .env (contém credenciais). Ele está no .gitignore.
- Com staging misto, use git add -A (ou adicione os caminhos explicitamente)
  antes de commitar.
- Não faça push sem solicitação.

## Estilo de trabalho
- Antes de implementar, diagnostique o que já existe e reporte — não recrie o
  que já está pronto.
- Implemente uma mudança de cada vez e pare para validação antes de avançar.
- Não corrija problemas silenciosamente: reporte e deixe a decisão para o autor.
