
from io import BytesIO
from pathlib import Path

# from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}
DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MiB


def load_uploaded_document(
    filename: str,
    file_bytes: bytes,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
) -> list[Document]:
    """
    Extract text from an uploaded PDF, TXT, or Markdown file.

    Args:
        filename: Original uploaded filename.
        file_bytes: Raw bytes returned by the upload interface.
        max_file_size: Maximum accepted file size in bytes.

    Returns:
        LangChain Documents with source and page metadata.
        PDFs produce one Document per page; text files produce one
        Document per file.

    Raises:
        ValueError: If the file is invalid or unsupported.
    """

    # 1. Validate the filename
    if not filename or not filename.strip():
        raise ValueError("Filename cannot be empty.")

    # Ignore any directory portion supplied in the filename.
    safe_filename = Path(filename.replace("\\", "/")).name
    extension = Path(safe_filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension or 'unknown'}. "
            f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    # 2. Validate the uploaded content
    if not isinstance(file_bytes, bytes) or not file_bytes:
        raise ValueError("Uploaded file is empty or invalid.")

    if max_file_size <= 0:
        raise ValueError("Maximum file size must be positive.")

    if len(file_bytes) > max_file_size:
        raise ValueError(
            f"File exceeds the {max_file_size}-byte upload limit."
        )

    # 3. Extract the text
    if extension == ".pdf":
        try:
            # Load the PDF from memory, without saving it to disk.
            import pypdf

            reader = pypdf.PdfReader(BytesIO(file_bytes))

            if reader.is_encrypted:
                raise ValueError("Encrypted PDFs are not supported.")

            documents = []

            for page_number, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""

                documents.append(
                    Document(
                        page_content=page_text,
                        metadata={"page": page_number},
                    )
                )

        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(
                f"Could not read PDF '{safe_filename}'."
            ) from exc

    else:
        try:
            text = file_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"'{safe_filename}' must contain UTF-8 text."
            ) from exc

        if not text.strip():
            raise ValueError(
                f"No text content found in '{safe_filename}'."
            )

        documents = [
            Document(page_content=text, metadata={})
        ]

    # 4. Attach source information to every Document
    for document in documents:
        document.metadata.update(
            {
                "source": safe_filename,
                "file_name": safe_filename,
                "file_type": extension.lstrip("."),
            }
        )

    # 5. Reject files that contain no extractable text
    if not any(doc.page_content.strip() for doc in documents):
        raise ValueError(
            f"No extractable text found in '{safe_filename}'. "
            "The PDF may be scanned or image-based."
        )

    return documents
