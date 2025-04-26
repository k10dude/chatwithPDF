import streamlit as st
import openai
import chromadb
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
import hashlib

# ------------------------------
# CONFIG
# ------------------------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
CHROMA_DB_DIR = "./chroma_db"

# Initialize ChromaDB
chroma_client = chromadb.Client()
if not os.path.exists(CHROMA_DB_DIR):
    os.makedirs(CHROMA_DB_DIR)
collection = chroma_client.get_or_create_collection(name="pdf_data")

# Keep track of uploaded PDFs
if "pdf_files" not in st.session_state:
    st.session_state.pdf_files = {}

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def generate_doc_id(text, filename):
    """Generate unique ID for a text chunk + filename using SHA256."""
    combined = filename + text
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()

def embed_and_store_text(texts, filename):
    """Embed text chunks if not already stored."""
    for text in texts:
        if not text.strip():  # Skip empty pages
            continue
        doc_id = generate_doc_id(text, filename)
        existing = collection.get(ids=[doc_id])
        if existing and existing['ids']:
            continue  # Already embedded, skip
        embedding = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )["data"][0]["embedding"]
        collection.add(
            embeddings=[embedding],
            documents=[text],
            ids=[doc_id],
            metadatas=[{"source": filename}]
        )

def retrieve_relevant_context(query, selected_pdf):
    """Embed query and retrieve matching chunks from the selected PDF only."""
    query_embedding = openai.Embedding.create(
        input=query,
        model="text-embedding-ada-002"
    )["data"][0]["embedding"]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where={"source": selected_pdf}
    )
    docs = results.get("documents", [])
    flattened = [doc for sublist in docs for doc in sublist if doc]
    context = "\n\n".join(flattened)
    return context[:4000]  # Limit context size

def ask_gpt4(query, context=""):
    """Ask GPT-4 using the retrieved context."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the given context when answering."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.2,
        max_tokens=800
    )
    return response.choices[0].message['content']

def extract_text_from_pdf(uploaded_pdf):
    """Extract and clean text from uploaded PDF."""
    reader = PdfReader(uploaded_pdf)
    texts = []
    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():  # Only keep non-empty pages
            texts.append(text)
    return texts

def summarize_text(full_text):
    """Ask GPT-4 to summarize the entire document."""
    prompt = f"Please summarize the following document:\n\n{full_text[:10000]}"  # limit to first ~4k tokens
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=800
    )
    return response.choices[0].message['content']

# ------------------------------
# STREAMLIT UI
# ------------------------------
st.set_page_config(page_title="💬 Chat with PDFs + GPT-4", page_icon="📄")
st.title("📄 Chat with your PDF Files + GPT-4")

# Sidebar Upload Section
with st.sidebar:
    uploaded_files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            texts = extract_text_from_pdf(uploaded_file)
            embed_and_store_text(texts, uploaded_file.name)
            full_text = " ".join(texts)
            summary = summarize_text(full_text)
            st.session_state.pdf_files[uploaded_file.name] = {
                "summary": summary,
                "full_text": full_text
            }
        st.success(f"{len(uploaded_files)} file(s) uploaded and processed!")

# Select which PDF to chat with
selected_pdf = None
if st.session_state.pdf_files:
    selected_pdf = st.selectbox("Select a PDF to chat with:", list(st.session_state.pdf_files.keys()))

    # Show Summary of Selected PDF
    st.subheader("📝 Document Summary")
    st.markdown(st.session_state.pdf_files[selected_pdf]["summary"])

# Session to maintain chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
query = st.chat_input("Ask a question about your selected document...")
if query and selected_pdf:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            context = retrieve_relevant_context(query, selected_pdf)
            answer = ask_gpt4(query, context)
            st.markdown(answer)

            # Show the context chunks (sources) under the answer
            if context.strip():
                with st.expander("📚 Sources (Context Used)"):
                    st.markdown(context)

    st.session_state.messages.append({"role": "assistant", "content": answer})