"""
rag_engine.py
- Embeds chunks with Gemini's embedding model
- Stores/searches them in an in-memory Chroma vector store
- Retrieves the most relevant chunks for a question
- Asks Gemini to answer using ONLY the retrieved chunks
"""

import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
from config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)

EMBED_MODEL = "models/text-embedding-004"
CHAT_MODEL = "gemini-2.0-flash"


class GeminiEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __call__(self, input):
        embeddings = []
        for text in input:
            result = genai.embed_content(
                model=EMBED_MODEL,
                content=text,
                task_type="retrieval_document",
            )
            embeddings.append(result["embedding"])
        return embeddings


def build_vector_store(chunks: list[str], collection_name: str = "pdf_collection"):
    """
    Creates a fresh in-memory Chroma collection from the given chunks.
    Re-run this whenever chunks change (new PDFs / new chunk settings).
    """
    client = chromadb.Client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=GeminiEmbeddingFunction(),
    )
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)
    return collection


def retrieve_chunks(collection, query: str, top_k: int = 4):
    """
    Returns a list of (chunk_text, distance) tuples, most relevant first.
    """
    result = genai.embed_content(
        model=EMBED_MODEL,
        content=query,
        task_type="retrieval_query",
    )
    query_embedding = result["embedding"]

    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    docs = results["documents"][0]
    distances = results["distances"][0]
    return list(zip(docs, distances))


def generate_answer(query: str, retrieved_chunks: list[tuple[str, float]]) -> str:
    context = "\n\n---\n\n".join(chunk for chunk, _ in retrieved_chunks)

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not contained in the context, say you don't know — do not make things up.

Context:
{context}

Question: {query}

Answer:"""

    model = genai.GenerativeModel(CHAT_MODEL)
    response = model.generate_content(prompt)
    return response.text