import os
import sys

# Add project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.clause_parser import parse_3gpp_pdf

DATA_DIR = "./data/3gpp_specs"
CHROMA_PATH = "./chroma_db"


def build_index():
    os.makedirs(DATA_DIR, exist_ok=True)
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))

    if not pdf_files:
        return False, "No PDF files found in data/3gpp_specs."

    all_chunks = []
    for pdf_path in pdf_files:
        chunks = parse_3gpp_pdf(pdf_path)
        all_chunks.extend(chunks)

    if not all_chunks:
        return False, "Failed to extract text chunks from uploaded PDFs."

    # 100% Free Local Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Persist in ChromaDB
    Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )
    return True, f"Successfully indexed {len(all_chunks)} chunks from {len(pdf_files)} PDF(s)!"


if __name__ == "__main__":
    success, msg = build_index()
    print(msg)