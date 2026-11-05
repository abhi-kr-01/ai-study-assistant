
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


# Locate the project's root directory.
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from the root .env file.
load_dotenv(BASE_DIR / ".env")


def get_chat_model() -> ChatGoogleGenerativeAI:
    """Initialize and return the Gemini chat model."""

    try:
        api_key = st.secrets.get("GOOGLE_GEMINI_API_KEY")
    except Exception:
        api_key = None

    api_key = api_key or os.getenv("GOOGLE_GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is missing. Check your root .env file."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0,
        google_api_key=api_key,
    )
