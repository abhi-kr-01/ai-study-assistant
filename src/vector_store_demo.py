
from langchain_core.documents import Document

from src.vector_store import index_documents, search_documents


def main():
    sample_chunks = [
        Document(
            page_content=(
                "Round Robin scheduling assigns each process "
                "a fixed time quantum in cyclic order."
            ),
            metadata={
                "source": "os-notes.pdf",
                "file_type": "pdf",
                "page": 2,
                "chunk_index": 0,
            },
        ),
        Document(
            page_content=(
                "A deadlock occurs when processes wait indefinitely "
                "for resources held by one another."
            ),
            metadata={
                "source": "os-notes.pdf",
                "file_type": "pdf",
                "page": 5,
                "chunk_index": 1,
            },
        ),
        Document(
            page_content=(
                "Virtual memory allows programs to use an address "
                "space larger than available physical memory."
            ),
            metadata={
                "source": "memory-notes.md",
                "file_type": "md",
                "chunk_index": 0,
            },
        ),
    ]

    ids = index_documents(sample_chunks)
    print(f"Indexed {len(ids)} chunks.")

    query = "How does the operating system share CPU time?"

    results = search_documents(query, k=2)

    print(f"\nQuestion: {query}")

    for rank, (document, distance) in enumerate(results, start=1):
        print(f"\nResult {rank}")
        print("Text:", document.page_content)
        print("Source:", document.metadata.get("source"))
        print("Page:", document.metadata.get("page", "N/A"))
        print("Distance:", round(distance, 4))


if __name__ == "__main__":
    main()

