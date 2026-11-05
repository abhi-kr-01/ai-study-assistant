
from io import BytesIO

import pytest
from pypdf import PdfWriter

from src.document_loader import load_uploaded_document


def make_test_pdf() -> bytes:
    """Create a tiny PDF in memory for testing."""
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)

    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def test_load_text_file():
    content = b"An operating system manages computer resources."

    documents = load_uploaded_document("notes.txt", content)

    assert len(documents) == 1
    assert "operating system" in documents[0].page_content
    assert documents[0].metadata["source"] == "notes.txt"
    assert documents[0].metadata["file_type"] == "txt"


def test_reject_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_uploaded_document("notes.docx", b"some content")


def test_reject_empty_file():
    with pytest.raises(ValueError, match="empty or invalid"):
        load_uploaded_document("notes.txt", b"")


def test_reject_oversized_file():
    with pytest.raises(ValueError, match="exceeds"):
        load_uploaded_document(
            "notes.txt",
            b"12345",
            max_file_size=4,
        )


def test_pdf_without_extractable_text():
    pdf_bytes = make_test_pdf()

    with pytest.raises(ValueError, match="No extractable text"):
        load_uploaded_document("notes.pdf", pdf_bytes)
