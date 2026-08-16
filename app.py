import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from scripts.build_index import build_index
from src.chain import create_rag_chain
from src.retriever import load_advanced_retriever

st.set_page_config(page_title="3GPP Telecom Assistant", page_icon="📡", layout="wide")

st.title("Grounded 3GPP Telecom RAG Assistant")
st.caption("Powered by Mistral AI, Hybrid Search & Guardrails")

with st.sidebar:
    st.header("Upload 3GPP Documents")
    uploaded_files = st.file_uploader("Upload 3GPP Spec PDFs", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        save_dir = "./data/3gpp_specs"
        os.makedirs(save_dir, exist_ok=True)
        for uploaded_file in uploaded_files:
            with open(os.path.join(save_dir, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded {len(uploaded_files)} file(s)!")

        if st.button("Build Vector Database Index"):
            with st.spinner("Parsing PDFs and building ChromaDB index..."):
                success, msg = build_index()
                if success:
                    st.success(msg)
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.error(msg)

if not os.getenv("MISTRAL_API_KEY"):
    st.warning(" `MISTRAL_API_KEY` missing in `.env` file!")

if not os.path.exists("./chroma_db") or not os.listdir("./chroma_db"):
    st.error("Vector database empty! Please upload a PDF in the sidebar and click 'Rebuild Vector Database Index'.")
    st.stop()

@st.cache_resource
def init_system():
    return create_rag_chain(load_advanced_retriever())

try:
    chain = init_system()
except Exception as e:
    st.error(f"System Initialization Error: {e}")
    st.stop()

query = st.text_input("Ask a 3GPP Question (e.g., 'What are the main functions of AMF in 5G NR?'):")

if query:
    with st.spinner("Retrieving standards & verifying answer grounding..."):
        try:
            res = chain({"question": query})
            ans = res["response"]

            st.subheader("Answer")
            st.write(ans.answer)

            col1, col2 = st.columns(2)
            col1.metric("Fully Grounded", "Yes" if ans.is_fully_grounded else "No")
            col2.metric("Confidence Score", f"{ans.confidence_score * 100:.1f}%")

            if ans.citations:
                st.markdown("**Citations:**")
                for c in ans.citations:
                    st.write(f"- `{c}`")

            with st.expander("Inspect Retrieved 3GPP Chunks", expanded=True):
                for idx, doc in enumerate(res["sources"]):
                    st.markdown(f"**Chunk {idx+1}** | `{doc.metadata.get('spec_number', '3GPP Spec')}`")
                    st.write(doc.page_content)
                    st.markdown("---")
        except Exception as e:
            st.error(f"Execution Error: {e}")