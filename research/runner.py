from research.trials_pipeline import (
    add_custom_document,
    ask_questions,
    build_text_chunks,
    build_vector_store,
    create_rag_chain,
    load_env,
)


def main() -> None:
    pinecone_api_key, _ = load_env()

    text_chunks = build_text_chunks(data_dir="data")
    docsearch = build_vector_store(
        pinecone_api_key=pinecone_api_key,
        texts_chunk=text_chunks,
        index_name="medical-chatbot",
    )

    add_custom_document(
        docsearch=docsearch,
        content="Medical Chatbot is a tool designed to provide information and answer questions related to medical topics. It can assist users in understanding various medical conditions, treatments, and general health information.",
        source="Youtube",
    )

    rag_chain = create_rag_chain(docsearch=docsearch, model="gpt-4o")

    questions = [
        "what is Acromegaly and gigantism?",
        "what is Acne?",
        "what is the Treatment of Acne?",
        "what is Cancer?",
    ]

    answers = ask_questions(rag_chain, questions)
    for question, answer in answers.items():
        print(f"Q: {question}")
        print(f"A: {answer}")
        print("-" * 80)


if __name__ == "__main__":
    main()
