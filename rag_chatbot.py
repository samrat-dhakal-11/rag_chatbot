"""
rag_chatbot.py
Main Streamlit app.

Run with:
    streamlit run rag_chatbot.py

Flow (matches the assignment's step1 / step2):
  Tab 1  -> Upload multiple PDFs, parse them with LlamaParse
  Tab 2  -> View the parsed PDF text
  Tab 3  -> Chunk the parsed text, view the chunks
  Tab 4  -> Ask a question -> see retrieved chunks -> get an answer from Gemini,
            grounded only in the uploaded PDF(s)
"""

import streamlit as st
from auth import login_ui, logout_button
from pdf_parser import parse_pdf
from chunker import chunk_text
from rag_engine import build_vector_store, retrieve_chunks, generate_answer

st.set_page_config(page_title="RAG PDF Chatbot", layout="wide")

# ---- LOGIN GATE ----
if not login_ui():
    st.stop()

st.sidebar.write(f"Logged in as: **{st.session_state.username}**")
logout_button()

st.title("📚 RAG PDF Chatbot")

# ---- SESSION STATE ----
if "parsed_docs" not in st.session_state:
    st.session_state.parsed_docs = {}      # {filename: parsed_text}
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []       # list[str]
if "collection" not in st.session_state:
    st.session_state.collection = None     # Chroma collection

tab1, tab2, tab3, tab4 = st.tabs(
    ["1️⃣ Upload & Parse", "2️⃣ Parsed Text", "3️⃣ Chunks", "4️⃣ Chat (RAG)"]
)

# ================= TAB 1: UPLOAD & PARSE =================
with tab1:
    st.header("Step 1 — Upload PDFs and parse with LlamaParse")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files and st.button("Parse PDFs"):
        for f in uploaded_files:
            if f.name not in st.session_state.parsed_docs:
                with st.spinner(f"Parsing '{f.name}' with LlamaParse..."):
                    try:
                        text = parse_pdf(f)
                        st.session_state.parsed_docs[f.name] = text
                        st.success(f"Parsed: {f.name}")
                    except Exception as e:
                        st.error(f"Failed to parse {f.name}: {e}")
            else:
                st.info(f"'{f.name}' was already parsed.")

    if st.session_state.parsed_docs:
        st.write("Parsed files so far:", list(st.session_state.parsed_docs.keys()))

# ================= TAB 2: VIEW PARSED TEXT =================
with tab2:
    st.header("Parsed PDF content")
    if not st.session_state.parsed_docs:
        st.warning("No PDFs parsed yet — go to Tab 1 first.")
    else:
        chosen = st.selectbox("Select a PDF to view", list(st.session_state.parsed_docs.keys()))
        st.text_area("Parsed text", st.session_state.parsed_docs[chosen], height=500)

# ================= TAB 3: CHUNKING =================
with tab3:
    st.header("Step 2 — Chunk the parsed text")
    chunk_size = st.slider("Chunk size (characters)", 300, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap (characters)", 0, 500, 150, step=50)

    if not st.session_state.parsed_docs:
        st.warning("No PDFs parsed yet — go to Tab 1 first.")
    else:
        if st.button("Generate chunks & build vector store"):
            all_chunks = []
            for fname, text in st.session_state.parsed_docs.items():
                for c in chunk_text(text, chunk_size, chunk_overlap):
                    all_chunks.append(f"[Source: {fname}]\n{c}")

            st.session_state.all_chunks = all_chunks
            with st.spinner("Embedding chunks and building vector store..."):
                st.session_state.collection = build_vector_store(all_chunks)
            st.success(f"Created {len(all_chunks)} chunks and indexed them.")

    if st.session_state.all_chunks:
        st.write(f"Total chunks: {len(st.session_state.all_chunks)}")
        for i, c in enumerate(st.session_state.all_chunks):
            with st.expander(f"Chunk {i + 1}"):
                st.write(c)

# ================= TAB 4: CHAT / RAG =================
with tab4:
    st.header("Step 3 — Ask questions about your PDFs")

    if st.session_state.collection is None:
        st.warning("Please parse PDFs (Tab 1) and generate chunks (Tab 3) first.")
    else:
        query = st.text_input("Your question")
        top_k = st.slider("Number of chunks to retrieve", 1, 10, 4)

        if st.button("Ask") and query:
            with st.spinner("Retrieving relevant chunks..."):
                retrieved = retrieve_chunks(st.session_state.collection, query, top_k)

            st.subheader("🔎 Retrieved chunks (used as context)")
            for i, (chunk, dist) in enumerate(retrieved):
                with st.expander(f"Retrieved chunk {i + 1}  (distance: {dist:.4f})"):
                    st.write(chunk)

            with st.spinner("Generating answer with Gemini..."):
                try:
                    answer = generate_answer(query, retrieved)
                except Exception as e:
                    answer = f"Error generating answer: {e}"

            st.subheader("💬 Answer")
            st.write(answer)