import os
import pdfplumber
from langchain_core.documents import Document
from src.clause_parser import extract_3gpp_metadata_and_tag


def parse_3gpp_pdf(pdf_path: str) -> list[Document]:
    """Parses a 3GPP PDF document, extracts tables into Markdown format,

    tags metadata (clause, spec number), and returns a list of LangChain
    Document objects.
    """
    documents = []

    # Extract filename safely across Windows and POSIX systems
    file_name = os.path.basename(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            tables = page.extract_tables()

            table_md = ""
            for table in tables:
                if not table:
                    continue

                # Clean newline characters within cells to maintain valid markdown table formatting
                clean_table = [
                    [cell.replace("\n", " ") if cell else "" for cell in row]
                    for row in table
                ]
                headers = clean_table[0]
                rows = clean_table[1:]

                header_line = "| " + " | ".join(headers) + " |"
                sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
                row_lines = ["| " + " | ".join(r) + " |" for r in rows]
                table_md += (
                    f"\n\n{header_line}\n{sep_line}\n"
                    + "\n".join(row_lines)
                    + "\n\n"
                )

            # Combine page text and Markdown tables
            raw_page_content = text + table_md

            # Extract 3GPP specific clause numbers and enrich metadata
            metadata, tagged_content = extract_3gpp_metadata_and_tag(
                file_name, raw_page_content
            )
            metadata["page"] = page_idx + 1

            documents.append(
                Document(page_content=tagged_content, metadata=metadata)
            )

    return documents