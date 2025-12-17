from langchain_datastax import AstraDBVectorStore
from .embedings import get_embeddings
from .config import (
    ASTRA_DB_API_ENDPOINT,
    ASTRA_DB_APPLICATION_TOKEN,
    ASTRA_DB_KEYSPACE
)

def get_vectorstore(collection_name="rag_documents"):
    embeddings = get_embeddings()

    return AstraDBVectorStore(
        embedding=embeddings,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
        namespace=ASTRA_DB_KEYSPACE,
        collection_name=collection_name
    )
