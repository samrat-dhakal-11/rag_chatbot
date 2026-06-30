import streamlit as st
import os
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, Settings, ChatPromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
import nest_asyncio

# Setup
nest_asyncio.apply()

# --- 1. Environment ---
ENV_PATH = "/Users/pratiksha/nwee/helloo_there/.env"
load_dotenv(dotenv_path=ENV_PATH)

os.environ["GOOGLE_API_KEY"] = os.getenv("api_key", "")
api_key = os.getenv("LLAMA_PARSE_KEY")

# --- 2. Configure Gemini 1.5 Flash ---
Settings.llm = Gemini(model="models/gemini-1.5-flash")
Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001")

st.set_page_config(page_title="Gemini 1.5 Flash RAG", layout="wide")
st.title("📄 Multi-PDF RAG Assistant (Flash 1.5)")

# --- 3. Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "engine" not in st.session_state:
    st.session_state.engine = None

# Sidebar for "New Chat"
if st.sidebar.button("New Chat"):
    st.session_state.messages = []
    st.rerun()

# --- 4. Initialization ---
if "parser" not in st.session_state:
    st.session_state["parser"] = LlamaParse(api_key=api_key, result_type="markdown")

# --- 5. File Upload & Processing ---
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files and st.button("Process Documents"):
    all_docs = []
    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())
        with st.spinner(f"Parsing {file.name}..."):
            docs = st.session_state["parser"].load_data(file.name)
            all_docs.extend(docs)
    
    with st.spinner("Creating Index..."):
        index = VectorStoreIndex.from_documents(all_docs)
        # ChatEngine ले history आफैं manage गर्छ
        st.session_state.engine = index.as_chat_engine(chat_mode="context", llm=Settings.llm)
        st.success("Ready to chat!")

# --- 6. Chat History Interface ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask a question:"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if st.session_state.engine:
        with st.chat_message("assistant"):
            response = st.session_state.engine.chat(query)
            st.markdown(str(response))
            st.session_state.messages.append({"role": "assistant", "content": str(response)})
    else:
        st.warning("Please upload and process PDFs first!")