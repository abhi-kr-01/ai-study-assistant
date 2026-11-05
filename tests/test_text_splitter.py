
import pytest
from langchain_core.documents import Document

from src.text_splitter import split_documents


def test_splits_long_document():
    document = Document(
        page_content="A" * 2500,
        metadata={"source": "notes.txt"},
    )

    chunks = split_documents(
        [document],
        chunk_size=1000,
        chunk_overlap=150,
    )

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 1000 for chunk in chunks)


def test_preserves_metadata():
    document = Document(
        page_content="Operating systems manage computer resources.",
        metadata={
            "source": "os.pdf",
            "page": 2,
            "file_type": "pdf",
        },
    )

    chunks = split_documents([document], chunk_size=100, chunk_overlap=10)

    assert chunks
    assert all(chunk.metadata["source"] == "os.pdf" for chunk in chunks)
    assert all(chunk.metadata["page"] == 2 for chunk in chunks)
    assert all(chunk.metadata["file_type"] == "pdf" for chunk in chunks)


def test_assigns_chunk_indices():
    document = Document(
        page_content="word " * 500,
        metadata={"source": "notes.txt"},
    )

    chunks = split_documents([document], chunk_size=100, chunk_overlap=10)

    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(
        range(len(chunks))
    )


def test_rejects_invalid_overlap():
    document = Document(page_content="Some notes.")

    with pytest.raises(ValueError, match="smaller than"):
        split_documents([document], chunk_size=100, chunk_overlap=100)


def test_rejects_empty_document_list():
    with pytest.raises(ValueError, match="empty document list"):
        split_documents([])
