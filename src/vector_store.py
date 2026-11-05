import hashlib
from langchain_chroma import Chroma
from pathlib import Path
from langchain_core.documents import Document

from src.embedding import get_embedding_model

CHROMA_DIRECTORY = Path("chroma_db")
COLLECTION_NAME = "study_notes"

def get_vector_store()-> Chroma :
    """Open or create the persistent Chroma collection."""

    embedding_model = get_embedding_model()

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DIRECTORY)
    )

def _make_chunk_id(document : Document) -> str :
    """Create a deterministic ID for a document chunk."""

    metadata = document.metadata

    identity = "\n".join(
        [
            str(metadata.get("source", "")),
            str(metadata.get("page", "")),
            str(metadata.get("chunk_index", "")),
            document.page_content,
        ]
    ) 

    return hashlib.sha256(identity.encode("utf-8")).hexdigest()

def index_documets(document: Document)->list[str]:
    """Embed and store document chunks in Chroma."""

    if not document :
        raise ValueError("Cannot index an empty document list.")

    if any(not doc.page_content.strip() for doc in document):
        raise ValueError("Cannot index empty document chunks.")

    vector_store = get_vector_store()

    ids = [_make_chunk_id(doc) for doc in document]

    # Reusing the same IDs avoids duplicate records when the same
    # chunks are indexed again.
    vector_store.add_documents(documents=document, ids=ids)

    return ids;    


def search_documets(
    query : str,
    k: int = 4,
    source : str | None = None,
) -> list[tuple[Document, float]]:
    """Return relevant chunks and their Chroma distance scores."""

    if not query.strip() :
        raise ValueError("Search query cannot be empty.")

    if(k < 1):
        raise ValueError("k must be at least 1.")

    vector_store = get_vector_store()

    metadata_filter = {"source": source} if source else None

    return vector_store.similarity_search_by_score(
        query=query,
        k = k,
        filter = metadata_filter
    )

