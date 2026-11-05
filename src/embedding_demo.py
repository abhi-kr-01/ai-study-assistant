from src.embedding import get_embedding_model

def main():
    embedding = get_embedding_model()

    text = (
        "CPU scheduling determines which process "
        "gets to execute on the CPU."
    )

    #Embed one peice of text 
    vector = embedding.embed_query(text)
    # vector = embedding.embed_documents(
    #     [
    #         "CPU scheduling selects a process to execute.",
    #         "Deadlocks can occur when processes wait indefinitely.",
    #     ]
    # )

    print("Input text:", text)
    print("Vector type:", type(vector))
    print("Vector dimensions:", len(vector))
    print("First 10 values:", vector[:10])


if __name__ == "__main__":
    main()

