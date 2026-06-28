-- Habilita a extensão de vetores usada pelo retrieval (pgvector).
-- Executado automaticamente pelo Postgres apenas na primeira inicialização
-- do volume de dados (diretório /docker-entrypoint-initdb.d/).
CREATE EXTENSION IF NOT EXISTS vector;
