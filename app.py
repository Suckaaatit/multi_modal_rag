import streamlit as st
import os
import shutil
from dotenv import load_dotenv

# FIX: ChromaDB requires sqlite3 >= 3.35. 
# This handles the error on older systems/Windows.
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from ingestion import PDFIngestionPipeline
from rag_engine import RAGEngine

st.set_page_config(page_title="RAG Chat", layout="wide")

if "current_file" not in st.session_state:
    st.session_state.current_file = None

# Load environment variables
load_dotenv()

# --- Sidebar UI ---
with st.sidebar:
    st.header("Setup")
    # Use API key from .env if available, otherwise show input field
    default_api_key = os.getenv('MISTRAL_API_KEY', '')
    api_key = st.text_input("Mistral API Key", 
                          value=default_api_key, 
                          type="password",
                          help="Get your API key from https://console.mistral.ai/")
    
    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
    
    if st.button("Reset"):
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
        st.session_state.current_file = None
        st.rerun()

# --- Main UI ---
st.title("📄 Document Intelligence")

if not api_key:
    st.warning("Please enter your Mistral API Key to continue.")
    st.stop()

try:
    pipeline = PDFIngestionPipeline(api_key)
    if 'engine' not in st.session_state:
        st.session_state.engine = RAGEngine(api_key)
except ValueError as e:
    st.error(f"API Key Error: {str(e)}")
    st.stop()
except Exception as e:
    st.error(f"Error initializing Mistral AI: {str(e)}")
    st.stop()

# 1. Ingestion Phase
if uploaded_file and st.session_state.current_file != uploaded_file.name:
    with st.spinner("Processing..."):
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        nodes = pipeline.process_pdf(file_bytes, uploaded_file.name)
        st.session_state.engine.index_documents(nodes, uploaded_file.name)
        st.session_state.current_file = uploaded_file.name
        st.success(f"Indexed {len(nodes)} chunks.")

# 2. Chat Phase
if st.session_state.current_file:
    # Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Ready."}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Input
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving & Reasoning..."):
                ans, sources = st.session_state.engine.query(prompt)
                st.write(ans)
                with st.expander("Sources"):
                    for n in sources:
                        st.caption(f"Page {n.metadata['page']}")
        st.session_state.messages.append({"role": "assistant", "content": ans})
