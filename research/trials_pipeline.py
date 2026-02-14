from __future__ import annotations

import os
from typing import Iterable, List

from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.helper import (
    download_hugging_face_embeddings,
    filter_to_minimal_docs,
    load_pdf_file,
    text_split,
)


def load_env() -> tuple[str, str]:
    load_dotenv()

    pinecone_api_key = os.getenv("PINECONE_API_KEY", "")
    openai_api_key = os.getenv("OPENAI_API_KEY", "")

    if not pinecone_api_key or not openai_api_key:
        raise ValueError("Set PINECONE_API_KEY and OPENAI_API_KEY in your environment or .env file.")

    os.environ["PINECONE_API_KEY"] = pinecone_api_key
    os.environ["OPENAI_API_KEY"] = openai_api_key

    return pinecone_api_key, openai_api_key


def build_text_chunks(data_dir: str = "data") -> List[Document]:
    extracted_data = load_pdf_file(data=data_dir)
    minimal_docs = filter_to_minimal_docs(extracted_data)
    return text_split(minimal_docs)


def ensure_pinecone_index(
    pinecone_client: Pinecone,
    index_name: str,
    dimension: int = 384,
    metric: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1",
) -> None:
    if not pinecone_client.has_index(index_name):
        pinecone_client.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )


def build_vector_store(
    pinecone_api_key: str,
    texts_chunk: List[Document],
    index_name: str = "medical-chatbot",
) -> PineconeVectorStore:
    embeddings = download_hugging_face_embeddings()
    pinecone_client = Pinecone(api_key=pinecone_api_key)

    ensure_pinecone_index(pinecone_client=pinecone_client, index_name=index_name)

    return PineconeVectorStore.from_documents(
        documents=texts_chunk,
        embedding=embeddings,
        index_name=index_name,
    )


def load_existing_vector_store(
    index_name: str = "medical-chatbot",
) -> PineconeVectorStore:
    embeddings = download_hugging_face_embeddings()
    return PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
    )


def create_rag_chain(docsearch: PineconeVectorStore, model: str = "gpt-4o"):
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    system_prompt = (
        "You are an Medical assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    chat_model = ChatOpenAI(model=model)
    question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


def add_custom_document(docsearch: PineconeVectorStore, content: str, source: str) -> None:
    doc = Document(page_content=content, metadata={"source": source})
    docsearch.add_documents(documents=[doc])


def ask_questions(rag_chain, questions: Iterable[str]) -> dict[str, str]:
    answers: dict[str, str] = {}
    for question in questions:
        response = rag_chain.invoke({"input": question})
        answers[question] = response["answer"]
    return answers
