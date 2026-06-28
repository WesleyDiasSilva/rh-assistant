"""Indexa as políticas (.md de fake_data/) na base vetorial.

Usa retrieval.carregar_docs() (leitura dos .md, com título extraído do
cabeçalho) e indexa cada documento via retrieval.indexar_documento.

Idempotente: indexar_documento usa IDs estáveis por arquivo/chunk, então rodar
o seed novamente faz upsert sobre os mesmos IDs e não duplica os chunks.
"""
from app.retrieval import carregar_docs, indexar_documento


def main() -> None:
    docs = carregar_docs()
    total_chunks = 0
    for arquivo, titulo, conteudo in docs:
        n = indexar_documento(conteudo, {"arquivo": arquivo, "titulo": titulo})
        total_chunks += n
        print(f"  indexado: {arquivo} ({titulo}) -> {n} chunks")
    print(f"Total: {len(docs)} arquivos, {total_chunks} chunks.")


if __name__ == "__main__":
    main()
