import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import ollama

# Page settings
st.set_page_config(page_title="Mini RAG", page_icon="📚")

st.title("📚 Mini RAG: Document Q&A")
st.write("PDF → Chunks → Embeddings → ChromaDB → Retrieval → Ollama")

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="documents"
)

# PDF upload
uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file:
    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    if not text.strip():
        st.error("The uploaded PDF is empty.")
        st.stop()

    # Split text into chunks
    chunk_size = 500

    chunks = [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size)
    ]

    st.write("Number of chunks:", len(chunks))

    # Create embeddings
    embeddings = embedding_model.encode(chunks).tolist()

    # Store in ChromaDB
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )

    st.success("PDF stored successfully!")

# Question
question = st.text_input("Ask a question about the PDF")

if question:

    # Convert question into embedding
    question_embedding = embedding_model.encode(
        [question]
    ).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=3
    )

    retrieved_chunks = results["documents"][0]

    # Combine retrieved information
    context = "\n\n".join(retrieved_chunks)

    # Prompt Ollama
    prompt = f"""
Answer the question using only the information given in the context.

Context:
{context}

Question:
{question}

Answer:
"""

    # Generate answer
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    st.subheader("Answer")
    st.write(answer)

    # Show retrieved chunks
    with st.expander("Retrieved Information"):
        for i, chunk in enumerate(retrieved_chunks):
            st.write(f"### Chunk {i + 1}")
            st.write(chunk)