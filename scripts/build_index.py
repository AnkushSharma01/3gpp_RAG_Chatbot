import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.table_ingester import parse_3gpp_pdf 

DATA_DIR = "./data/3gpp_specs"
CHROMA_PATH = "./chroma_db"


def build_index():
    os.makedirs(DATA_DIR, exist_ok=True)
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))

    if not pdf_files:
        return False, "No PDF files found in data/3gpp_specs."

    all_pages = []
    for pdf_path in pdf_files:
        pages = parse_3gpp_pdf(pdf_path)
        all_pages.extend(pages)

    if not all_pages:
        return False, "Failed to extract text chunks from uploaded PDFs."

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    all_chunks = text_splitter.split_documents(all_pages)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )
    return True, f"Successfully indexed {len(all_chunks)} chunks from {len(pdf_files)} PDF(s)!"


if __name__ == "__main__":
    success, msg = build_index()
    print(msg)