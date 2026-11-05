import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pathlib import Path
from src.document_loader import load_uploaded_document
from src.text_splitter import split_documents


load_dotenv()

# env_path = Path(__file__).resolve().parent.parent / ".env"
# load_dotenv(env_path, override=True)

# def main():

#     api_key = os.getenv("GOOGLE_GEMINI_API_KEY")

#     if not api_key : 
#         raise ValueError("GOOGLE_API_KEY is missing from .env")

#     #Check that the API key exists before calling Gemini
#     if not os.getenv('GOOGLE_GEMINI_API_KEY'):
#         raise ValueError("GOOGLE_API_KEY is missing : Add it your in .env file")


#     # create gemini model 
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-3.8-flash",
#         google_api_key=api_key,
#         temperature=0.2, #favors relatively consistent answers rather than highly varied outputs. It does not guarantee factual correctness.
#         max_retries=2 #allows retries for certain transient failures.
#     )

#     # send a question
#     question = ("Explain vector embeddings to a computer science student "
#         "using a simple example.")

#     response = llm.invoke(question)

#     #print
#     print("\nGemini's response:\n")
#     print(response.content)

def main() :
    path = Path("data/raw/10. Learning Roadmap.pdf")
    files_bytes = path.read_bytes()

    documents = load_uploaded_document(filename=path.name,file_bytes=files_bytes)

    chunks = split_documents(documents)

    print(f"Pages/documets loaded: {len(documents)}")
    print(f"Chunks created : {len(chunks)}")

    for chunk in chunks[:3]:
        print("\n Metadata: ",chunk.metadata)
        print("Text preveiw : ",chunk.page_content[:200])


if __name__ == "__main__":
    main()