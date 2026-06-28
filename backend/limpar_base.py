"""Remove todos os chunks da base vetorial (collection de políticas).

Útil para recriar a base do zero antes de um novo seed. Idempotente: rodar com
a base já vazia (ou nunca inicializada) não causa erro.
"""
from app.retrieval import COLLECTION_NAME, contar_chunks, get_vectorstore


def main() -> None:
    # Conta antes de remover, para reportar quantos chunks havia.
    antes = contar_chunks()
    # delete_collection() apaga a collection e seus embeddings; um seed posterior
    # recria a collection vazia automaticamente.
    get_vectorstore().delete_collection()
    print(f"Removidos {antes} chunks da collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
