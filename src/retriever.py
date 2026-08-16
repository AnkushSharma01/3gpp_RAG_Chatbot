import os
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_PATH = "./chroma_db"


def load_advanced_retriever():
    """Loads ChromaDB and BM25 retrievers using local HuggingFace embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=embeddings
    )
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    all_docs = vectorstore.get()
    documents = (
        [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
        ]
        if all_docs and "documents" in all_docs
        else []
    )

    bm25_retriever = (
        BM25Retriever.from_documents(documents) if documents else None
    )
    if bm25_retriever:
        bm25_retriever.k = 5

    return dense_retriever, bm25_retriever