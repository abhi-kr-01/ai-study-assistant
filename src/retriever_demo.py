
from langchain_core.documents import Document

from src.vector_store import index_documets
from src.retriever import retriever_documet


def main():
    # Sample study notes
    documents = [
        Document(
            page_content=(
                "Round Robin scheduling gives every process "
                "a fixed time quantum in cyclic order."
            ),
            metadata={
                "source": "retriever-os.pdf",
                "page": 1,
                "chunk_index": 0,
                "file_type": "pdf",
            },
        ),
        Document(
            page_content=(
                "FCFS scheduling executes processes in the order "
                "they arrive in the ready queue."
            ),
            metadata={
                "source": "retriever-os.pdf",
                "page": 2,
                "chunk_index": 1,
                "file_type": "pdf",
            },
        ),
        Document(
            page_content=(
                "Deadlock occurs when processes wait indefinitely "
                "for resources held by other processes."
            ),
            metadata={
                "source": "retriever-os.pdf",
                "page": 5,
                "chunk_index": 2,
                "file_type": "pdf",
            },
        ),
    ]

    # Index sample documents
    index_documets(documents)

    # Retrieve the top 2 relevant chunks
    query = "How does the operating system distribute CPU time?"

    results = retriever_documet(query, k=2)

    print(f"\nQuestion: {query}")
    print(f"Retrieved documents: {len(results)}")

    for index, document in enumerate(results, start=1):
        print(f"\nResult {index}")
        print("Content:", document.page_content)
        print("Metadata:", document.metadata)

    # Test metadata filtering
    print("\n--- Source-filtered retrieval ---")

    filtered_results = retriever_documet(
        query="What is deadlock?",
        k=3,
        source="retriever-os.pdf",
    )

    for document in filtered_results:
        print(
            document.metadata["source"],
            document.metadata.get("page"),
            document.page_content,
        )


if __name__ == "__main__":
    main()
