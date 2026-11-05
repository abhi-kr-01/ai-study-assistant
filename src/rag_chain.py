
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
)

from src.llm import get_chat_model
from src.vector_store import get_vector_store
from src.prompts import rag_prompt
from src.formatters import format_documents


def create_rag_chain():

    # 1. Initialize Gemini chat model
    llm = get_chat_model()

    # 2. Create retriever
    vector_store = get_vector_store()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    # 3. Construct RAG pipeline
    rag_chain = (
        RunnablePassthrough.assign(
            context=(
                RunnableLambda(lambda x: x["question"])
                | retriever
                | RunnableLambda(format_documents)
            )
        )
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
