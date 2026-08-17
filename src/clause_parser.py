import os
import re
import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CLAUSE_PATTERN = re.compile(r"\b\d{1,2}(?:\.\d{1,3}){1,4}\b")


def extract_3gpp_metadata_and_tag(file_name: str, raw_page_content: str):
    """
    Detects the primary 3GPP clause number present on a page and tags
    the content + metadata accordingly. Falls back gracefully if no
    clause number pattern is found (e.g. cover pages, TOC).
    """
    spec_number = file_name.replace(".pdf", "")

    matches = CLAUSE_PATTERN.findall(raw_page_content)
    clause_number = matches[0] if matches else "N/A"

    metadata = {
        "source": file_name,
        "spec_number": spec_number,
        "clause": clause_number,
    }

    tagged_content = f"[Spec: {spec_number} | Clause: {clause_number}]\n{raw_page_content}"

    return metadata, tagged_content


def parse_3gpp_pdf(pdf_path: str):
    """Legacy simple parser (page extract + generic char-splitting).
    Kept for reference/fallback use — build_index.py now uses
    table_ingester.parse_3gpp_pdf() as the primary path."""
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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    return text_splitter.split_documents(docs)