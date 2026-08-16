# 3GPP Telecom RAG Chatbot

An enterprise-grade Retrieval-Augmented Generation (RAG) system engineered to answer complex 3GPP telecom specification queries with strict context grounding and near-zero hallucinations.

## Live Application
* **Streamlit App:** https://3gppragchatbot-3ugjvyqesovddnsus37zb4.streamlit.app/

## System Architecture & Workflow
The system utilizes a hybrid retrieval pipeline tailored for dense technical documentation:

* **Language Model:** `Mistral-small-latest` via Mistral API for high-reasoning context synthesis.
* **Hybrid Retrieval Engine:** Combines dense semantic search via **ChromaDB** with sparse keyword search via **BM25** to accurately extract 3GPP acronyms, spec numbers, and protocols.
* **Embeddings:** Local `all-MiniLM-L6-v2` Sentence Transformers for low-latency vector encoding.
* **Pydantic Guardrails:** Custom validation layer (`enforce_guardrails`) that cross-examines generated responses directly against retrieved source chunks to suppress ungrounded outputs.

##  Hallucination Mitigation Strategy

1. **Dense + Sparse Fusion:** Prevents context retrieval loss when dealing with highly specific telecom acronyms (e.g., AMF, gNB, UPF, PDCP).
2. **Context Grounding Validation:** Outputs are parsed and structured through Pydantic schemas; any assertion lacking direct text support is filtered out.
3. **Automated Evaluation & Observability:** Measured using the **RAGAS** framework for faithfulness and answer relevance, with full pipeline tracing enabled via **LangSmith**.

##  Project Architecture & Directory Structure

```text
3gpp-telecom-rag/
├── chroma_db/                # Local ChromaDB persistent vector index
├── data/
│   ├── 3gpp_specs/           # Raw PDF specs (3GPP sample specifications)
│   └── 3gpp_acronyms.json    # Telecom domain acronym mappings (AMF, gNB, UPF, etc.)
├── scripts/
│   └── build_index.py        # Offline index generation and vector store setup
├── src/
│   ├── __init__.py
│   ├── acronym_expander.py   # Query expansion using domain acronym dictionary
│   ├── chain.py              # LangChain RAG synthesis & pipeline logic
│   ├── clause_parser.py      # Structural PDF parsing targeting 3GPP clause numbering
│   ├── guardrails.py         # Pydantic-enforced grounding and hallucination checks
│   ├── retriever.py          # Hybrid Retriever (ChromaDB + BM25 keyword search)
│   ├── schemas.py            # Structured Pydantic data schemas
│   └── table_ingester.py     # Special parsing pipeline for complex spec tables
├── app.py                    # Interactive Streamlit UI application
├── eval_ragas.py             # Evaluation script using the RAGAS framework
└── requirements.txt          # Production dependencies

##  Quickstart & Local Setup

```bash
# Clone the repository
git clone [https://github.com/AnkushSharma01/3gpp_RAG_Chatbot.git]

# Set up virtual environment
python -m venv venv
source venv/bin/activate  

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py