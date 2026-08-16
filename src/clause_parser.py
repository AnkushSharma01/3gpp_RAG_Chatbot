import os
import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def parse_3gpp_pdf(pdf_path: str):
    """Parses a 3GPP PDF file using pdfplumber and returns chunked LangChain Documents."""
    if not os.path.exists(pdf_path):
        return []

    docs = []
    file_name = os.path.basename(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                docs.append(
                    Document(
                        page_content=text.strip(),
                        metadata={
                            "source": file_name,
                            "page": page_num,
                            "spec_number": file_name.replace(".pdf", ""),
                        },
                    )
                )

    if not docs:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=100
    )
    chunks = text_splitter.split_documents(docs)

    return chunks