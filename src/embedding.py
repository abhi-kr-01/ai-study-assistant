import os 
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings  #LangChain provides GoogleGenerativeAIEmbeddings through the langchain-google-genai integration. It supports embedding documents and individual queries
# chromaDB is a vector db that store embedding vector or support similarity search and it also store original text and metadata alongside the vectors

def get_embedding_model() -> GoogleGenerativeAIEmbeddings :
    """Create the Google embedding model."""

    load_dotenv()

    try:
        api_key = st.secrets.get("GOOGLE_GEMINI_API_KEY")
    except Exception:
        api_key = None

    api_key = api_key or os.getenv("GOOGLE_GEMINI_API_KEY")

    if not api_key :
        raise ValueError("GOOGLE_API_KEY is missing from .env")

    if not os.getenv("GOOGLE_GEMINI_API_KEY"):
        raise ValueError("GOOGLE_API_KEY is missing from your .env file.")

    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        google_api_key=api_key
    )