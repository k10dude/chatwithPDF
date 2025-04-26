import streamlit as st
import openai
import faiss
import numpy as np
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
import hashlib

# ------------------------------
# CONFIG
# ------------------------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize FAISS
embedding_size = 1536  # OpenAI embedding dimension
index = faiss.IndexFlatL2(embedding_size)  # FAISS index
stored_texts = []  # to hold actual text chunks
metadata = []  # to hold filenames etc.

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def get_embedding(text):
    """Generate embedding using OpenAI."""
    result = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(result['data'][0]['embedding'], dtype="float32")

def embed_and_store_text(texts, filename):
    """Embed text chunks and add to FAISS index with metadata."""
    for text in texts:
        if not text.strip():
            continue  # skip empty
        embedding = get_embedding(text)
        index.add(np.array([embedding]))  # add to FAISS
        stored_texts.append(text)
        metadata.append({"source": filename})

def retrieve_relevant_context(query, selected_pdf):
    """Retrieve relevant chunks matching the selected PDF."""
    query_embedding = get_embedding(query)
    D, I = index.search(np.array([query_embedding]), k=5)  # 5 best chunks
    results = []
    for idx in I[0]:
        if idx < len(stored_texts) and metadata[idx]["source"] == selected_pdf:
            results.append(stored_texts[idx])
    context = "\n\n".join(results)
    return context[:4000]  # limit

def ask_gpt4(query, context=""):
    """Ask GPT-4 using the context."""
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
        if text and text.strip():
            texts.append(text)
    return texts

def summarize_text(full_text):
    """Summarize the entire document."""
    prompt = f"Please summarize the following document:\n\n{full_text[:10000]}"
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

# Session states
if "pdf_files" not in st.session_state:
    st.session_state.pdf_files = {}

if "messages" not in st.session_state:
    st.session_state.messages = []

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

            if context.strip():
                with st.expander("📚 Sources (Context Used)"):
                    st.markdown(context)

    st.session_state.messages.append({"role": "assistant", "content": answer})
