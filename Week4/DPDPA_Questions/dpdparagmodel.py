import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from io import BytesIO
from PyPDF2 import PdfReader  # make sure PyPDF2 is installed: pip install pypdf2

@st.cache_resource
def load_models():
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    llm = pipeline(
        "text-generation",
        model="distilgpt2",
        max_new_tokens=150
    )
    return embed_model, llm

embed_model, llm = load_models()

# -----------------------------
# 0) PDF utils
# -----------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts concatenated text from a PDF bytes buffer."""
    reader = PdfReader(BytesIO(file_bytes))
    texts = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        texts.append(txt)
    return "\n".join(texts).strip()

def simple_chunk(text: str, max_chars: int = 500, overlap: int = 100):
    """
    Simple character-based chunker to keep chunks within embedding limits.
    Overlap helps with context continuity.
    """
    text = text.replace("\r", " ")
    text = " ".join(text.split())  # normalize whitespace
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(text[start:end])
        start = end - overlap if end - overlap > start else end
    return [c for c in chunks if c.strip()]

# -----------------------------
# 1) Base KB
# -----------------------------
base_docs = [
    "LangChain helps developers build applications using large language models.",
    "RAG improves LLM accuracy by retrieving relevant external information.",
    "Sentence embeddings capture semantic meaning of text.",
    "Transformers use attention mechanisms to process language.",
    "Vector databases store embeddings for fast similarity search."
]

# We’ll store docs & embeddings in session_state to allow dynamic updates
if "docs" not in st.session_state:
    st.session_state.docs = base_docs.copy()
    st.session_state.doc_embeddings = embed_model.encode(st.session_state.docs)

# -----------------------------
# 2) Retriever
# -----------------------------
def retrieve(query, top_k=2):
    query_emb = embed_model.encode([query])[0]
    scores = np.dot(st.session_state.doc_embeddings, query_emb)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [st.session_state.docs[i] for i in top_indices]

# -----------------------------
# 3) RAG Answer
# -----------------------------
def generate_answer(query):
    retrieved_docs = retrieve(query)
    context = "\n".join(retrieved_docs)
    prompt = f"""
You are an AI assistant.
Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}

Answer:
"""
    response = llm(prompt)[0]["generated_text"]
    return response.split("Answer:")[-1].strip(), retrieved_docs

# -----------------------------
# 4) UI
# -----------------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 RAG Chatbot for DPDPA queries")
st.write("Ask questions based on the internal knowledge base. You can also upload a PDF to augment the KB.")

# ---- Upload a PDF and add to docs
pdf_file = st.file_uploader("📄 Upload a PDF to add to the knowledge base", type=["pdf"])
max_chars = st.slider("Chunk size (characters)", 200, 1500, 600, 50)
overlap = st.slider("Chunk overlap (characters)", 0, 300, 100, 10)

if pdf_file is not None:
    with st.spinner("Reading and chunking PDF..."):
        pdf_text = extract_text_from_pdf(pdf_file.read())
        if not pdf_text.strip():
            st.warning("Could not extract text from this PDF. Try another file.")
        else:
            chunks = simple_chunk(pdf_text, max_chars=max_chars, overlap=overlap)
            if not chunks:
                st.warning("No text chunks created from the PDF.")
            else:
                # Append to docs and re-embed incrementally
                st.session_state.docs.extend(chunks)
                new_embs = embed_model.encode(chunks)
                st.session_state.doc_embeddings = np.vstack([st.session_state.doc_embeddings, new_embs])
                st.success(f"Added {len(chunks)} chunks from the uploaded PDF to the knowledge base.")

user_query = st.text_input("Enter your question:")

if user_query:
    with st.spinner("Thinking..."):
        answer, sources = generate_answer(user_query)

    st.subheader("🧠 Answer")
    st.write(answer)

    st.subheader("📚 Retrieved Context")
    for src in sources:
        st.markdown(f"- {src}")