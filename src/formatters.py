
from langchain_core.documents import Document


def format_documents(documents: list[Document]) -> str:
    """Convert retrieved documents into readable context."""

    formatted_documents = []

    for document in documents:
        metadata = document.metadata

        source = metadata.get("source", "Unknown source")
        page = metadata.get("page")

        reference = f"Source: {source}"

        if page is not None:
            # PDF page metadata is zero-indexed.
            reference += f", Page: {page + 1}"

        formatted_documents.append(
            f"[{reference}]\n{document.page_content}"
        )

    return "\n\n".join(formatted_documents)
