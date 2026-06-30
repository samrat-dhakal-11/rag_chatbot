import streamlit as st
import os
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex
import nest_asyncio

# Setup
nest_asyncio.apply()

# --- 1. API Key Load गर्ने भाग ---
# यहाँ तपाईंको .env फाइलको पूर्ण Path दिनुहोस्
ENV_PATH = "/Users/pratiksha/nwee/helloo_there/.env"
load_dotenv(dotenv_path=ENV_PATH)

# Environment variable लिने
api_key = os.getenv("LLAMA_PARSE_KEY")

st.set_page_config(page_title="Multi-PDF RAG", layout="wide")
st.title("📄 Multi-PDF RAG Assistant")

# --- 2. LlamaParse को लागि Error handling ---
if not api_key:
    st.error(f"API Key भेटिएन! कृपया {ENV_PATH} मा 'LLAMA_PARSE_KEY' सहि छ कि छैन चेक गर्नुहोस्।")
    st.stop()

if "parser" not in st.session_state:
    st.session_state["parser"] = LlamaParse(
        api_key=api_key, 
        result_type="markdown"
    )

# --- 3. File Upload ---
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Process Documents"):
        all_docs = []
        for file in uploaded_files:
            # Temporary save
            file_path = f"./{file.name}"
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            
            # Parsing
            with st.spinner(f"Parsing {file.name}..."):
                try:
                    docs = st.session_state["parser"].load_data(file_path)
                    all_docs.extend(docs)
                    st.write(f"Parsed {file.name}")
                except Exception as e:
                    st.error(f"Error parsing {file.name}: {e}")
        
        # 4. Indexing
        if all_docs:
            with st.spinner("Creating Index..."):
                index = VectorStoreIndex.from_documents(all_docs)
                st.session_state["engine"] = index.as_query_engine()
                st.session_state["docs"] = all_docs
                st.success("Documents processed! You can now chat.")

# --- 5. Chat Interface ---
if "engine" in st.session_state:
    query = st.chat_input("Ask a question about your uploaded PDFs:")
    if query:
        st.chat_message("user").write(query)
        with st.spinner("Thinking..."):
            response = st.session_state["engine"].query(query)
            st.chat_message("assistant").write(str(response))
            
    # View Parsed content (Optional)
    if st.checkbox("Show Parsed Content"):
        for doc in st.session_state["docs"]:
            st.markdown("### Parsed Chunk")
            st.markdown(doc.text[:500] + "...")