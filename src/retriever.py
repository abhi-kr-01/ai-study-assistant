from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.vector_store import get_vector_store

DEFAULT_TOK_K = 4

def create_retriever(k : int = DEFAULT_TOK_K, source : str | None = None) -> BaseRetriever :
    """Create a LangChain retriever backed by ChromaDB.

    Args:
        k: Number of relevant chunks to retrieve.
        source: Optional filename to restrict the search.

    Returns:
        A LangChain retriever.
    """

    if k < 1 :
        raise ValueError("k must be at least 1.")

    vector_store = get_vector_store()

    search_kwargs = {"k":k}

    # Apply metadata filtering when a source is provided
    if source : 
        search_kwargs["filter"] = {"source": source}

    retriever = vector_store.as_retriever(search_type = "similarity",search_kwargs=search_kwargs)

    return retriever


def retriever_documet(query: str, k : int = DEFAULT_TOK_K, source : str | None = None)->list[Document]:
    """Retrieve relevant documents for a user question."""

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    retriever = create_retriever(k,source)

    return retriever.invoke(query)
