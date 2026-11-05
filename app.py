
import hashlib

import streamlit as st

from src.document_loader import load_uploaded_document
from src.text_splitter import split_documents
from src.vector_store import index_documets
from src.rag_chain import create_rag_chain


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 AI Study Assistant")
st.caption("Upload your notes and ask questions using AI.")


# --------------------------------------------------
# Session state initialization
# --------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processed_file_hash" not in st.session_state:
    st.session_state.processed_file_hash = None

if "processed_filename" not in st.session_state:
    st.session_state.processed_filename = None


# --------------------------------------------------
# Initialize RAG chain once per Streamlit session
# --------------------------------------------------

@st.cache_resource
def get_rag_chain():
    return create_rag_chain()


# --------------------------------------------------
# Sidebar: Document upload
# --------------------------------------------------

with st.sidebar:
    st.header("Upload your notes")

    uploaded_file = st.file_uploader(
        "Choose a document",
        type=["pdf", "txt", "md", "markdown"],
    )

    if uploaded_file is not None:

        file_bytes = uploaded_file.getvalue()

        file_hash = hashlib.sha256(file_bytes).hexdigest()

        st.write(f"**Selected:** {uploaded_file.name}")

        if st.button("Process document", type="primary"):

            try:
                with st.spinner("Processing your document..."):

                    documents = load_uploaded_document(
                        filename=uploaded_file.name,
                        file_bytes=file_bytes,
                    )

                    chunks = split_documents(documents)

                    index_documets(chunks)

                    st.session_state.processed_file_hash = file_hash
                    st.session_state.processed_filename = (
                        uploaded_file.name
                    )

                    # Clear old conversation after switching documents.
                    st.session_state.chat_history = []

                st.success(
                    f"Successfully processed {len(chunks)} chunks!"
                )

            except Exception as exc:
                st.error(f"Document processing failed: {exc}")

    if st.session_state.processed_filename:
        st.divider()
        st.success(
            f"Active document: {st.session_state.processed_filename}"
        )


# --------------------------------------------------
# Main chat interface
# --------------------------------------------------

if st.session_state.processed_file_hash is None:

    st.info("Upload and process a document to start chatting.")

else:

    st.subheader("Chat with your notes")

    # Display previous messages.
    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept a new question.
    question = st.chat_input("Ask something about your notes...")

    if question:

        # Display user message.
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        # Generate AI answer.
        with st.chat_message("assistant"):

            with st.spinner("Searching your notes..."):

                try:
                    rag_chain = get_rag_chain()

                    answer = rag_chain.invoke(
                        {"question": question}
                    )

                    st.markdown(answer)

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )

                except Exception as exc:
                    st.error(f"Could not generate answer: {exc}")
