
from src.rag_chain import create_rag_chain


def main():

    rag_chain = create_rag_chain()

    question = "How does Round Robin scheduling work?"

    response = rag_chain.invoke(
        {"question": question}
    )

    print("\nQuestion:")
    print(question)

    print("\nAI Answer:")
    print(response)


if __name__ == "__main__":
    main()
