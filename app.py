import streamlit as st
import os
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
import nest_asyncio

nest_asyncio.apply()

# --- 1. Environment ---
load_dotenv(dotenv_path="/Users/pratiksha/nwee/helloo_there/.env")
os.environ["GOOGLE_API_KEY"] = os.getenv("api_key", "")
llama_key = os.getenv("LLAMA_PARSE_KEY")

# --- 2. Settings ---
Settings.llm = Gemini(model="models/gemini-3.5-flash")
Settings.embed_model = GeminiEmbedding(model_name="models/gemini-embedding-001")
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

st.set_page_config(page_title="Advanced RAG Debugger", layout="wide")
st.title("📂 Advanced PDF RAG Pipeline")

# --- 3. Session State ---
if "engine" not in st.session_state: st.session_state.engine = None
if "docs_data" not in st.session_state: st.session_state.docs_data = {} # Parsed data store

# --- 4. Sidebar ---
if st.sidebar.button("New Chat"):
    st.session_state.clear()
    st.rerun()

# --- 5. File Processing ---
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if uploaded_files and st.button("Process & Parse"):
    all_nodes = []
    parser = LlamaParse(api_key=llama_key, result_type="markdown")
    
    for file in uploaded_files:
        with open(file.name, "wb") as f: f.write(file.getbuffer())
        with st.spinner(f"Parsing {file.name}..."):
            docs = parser.load_data(file.name)
            st.session_state.docs_data[file.name] = docs[0].text # Store parsed text
            nodes = Settings.node_parser.get_nodes_from_documents(docs)
            all_nodes.extend(nodes)
            
    index = VectorStoreIndex(all_nodes)
    st.session_state.engine = index.as_chat_engine(chat_mode="context", llm=Settings.llm)
    st.session_state.all_nodes = all_nodes
    st.success("Documents Ready!")

# --- 6. UI for Debugging (Parsed & Chunks) ---
col1, col2 = st.columns(2)
with col1:
    if st.session_state.docs_data:
        selected_file = st.selectbox("View Parsed PDF", list(st.session_state.docs_data.keys()))
        st.text_area("Parsed Markdown:", st.session_state.docs_data[selected_file], height=300)

with col2:
    if "all_nodes" in st.session_state:
        chunk_idx = st.slider("View Chunks", 0, len(st.session_state.all_nodes)-1, 0)
        st.info(f"Chunk {chunk_idx}:")
        st.write(st.session_state.all_nodes[chunk_idx].text)

# --- 7. Chat ---
query = st.chat_input("Ask a question:")
if query and st.session_state.engine:
    with st.chat_message("user"): st.markdown(query)
    # Retrieving for Debugging
    retriever = st.session_state.engine.retriever
    nodes = retriever.retrieve(query)
    
    with st.expander("View Retrieved Chunks (Context)"):
        for n in nodes: st.write(f"--- \n{n.node.text}")
            
    response = st.session_state.engine.chat(query)
    with st.chat_message("assistant"): st.markdown(str(response))