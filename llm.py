# File: /Users/ketan/Downloads/netsuite_jobs.py
import streamlit as st
import openai
import faiss
import numpy as np
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
import hashlib
import csv # Added for CSV processing
import io  # Added for CSV processing

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
    try:
        # --- DEBUG ---
        # st.write(f"DEBUG: Getting embedding for text chunk starting with: {text[:50]}...")
        # --- END DEBUG ---
        result = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        # --- DEBUG ---
        # st.write("DEBUG: Embedding received.")
        # --- END DEBUG ---
        return np.array(result['data'][0]['embedding'], dtype="float32")
    except Exception as e:
        st.error(f"Error getting embedding: {e}")
        return None # Handle embedding errors

def embed_and_store_text(texts, filename):
    """Embed text chunks and add to FAISS index with metadata."""
    # --- DEBUG ---
    st.write(f"DEBUG: Starting embed_and_store_text for {filename}. Number of text chunks: {len(texts)}")
    # --- END DEBUG ---
    embeddings_to_add = []
    texts_to_store = []
    metadata_to_add = []
    processed_count = 0 # Debug counter

    for i, text in enumerate(texts): # Add index for debugging
        if not text or not text.strip():
            continue  # skip empty or whitespace-only text

        # --- DEBUG ---
        if i % 50 == 0: # Log progress every 50 chunks
             st.write(f"DEBUG: Processing chunk {i+1}/{len(texts)} for embedding...")
        # --- END DEBUG ---
        embedding = get_embedding(text)
        if embedding is not None: # Check if embedding was successful
            embeddings_to_add.append(embedding)
            texts_to_store.append(text)
            metadata_to_add.append({"source": filename})
            processed_count += 1
        # --- DEBUG ---
        # else:
        #     st.write(f"DEBUG: Skipping chunk {i+1} due to embedding error or empty content.")
        # --- END DEBUG ---


    # --- DEBUG ---
    st.write(f"DEBUG: Finished embedding loop. {processed_count} chunks embedded.")
    # --- END DEBUG ---

    if embeddings_to_add:
        # --- DEBUG ---
        st.write(f"DEBUG: Adding {len(embeddings_to_add)} embeddings to FAISS index...")
        # --- END DEBUG ---
        index.add(np.array(embeddings_to_add))  # add batch to FAISS
        stored_texts.extend(texts_to_store)
        metadata.extend(metadata_to_add)
        st.write(f"Added {len(embeddings_to_add)} text chunks from {filename}.") # Keep this info message
        # --- DEBUG ---
        st.write("DEBUG: Embeddings added to FAISS.")
        # --- END DEBUG ---
    else:
        st.warning(f"No valid text chunks found or embedded for {filename}.")


def retrieve_relevant_context(query, selected_pdf):
    """Retrieve relevant chunks matching the selected PDF."""
    if index.ntotal == 0:
        st.warning("No documents have been indexed yet. Please upload files.")
        return ""
    query_embedding = get_embedding(query)
    if query_embedding is None:
        return "" # Handle embedding error for query

    try:
        D, I = index.search(np.array([query_embedding]), k=min(5, index.ntotal)) # Ensure k is not > index size
        results = []
        seen_indices = set() # Avoid duplicate indices if k > actual relevant results
        for idx in I[0]:
            # Check index bounds and avoid duplicates
            if idx >= 0 and idx < len(stored_texts) and idx not in seen_indices:
                 # Check if the metadata source matches the selected file
                 if metadata[idx]["source"] == selected_pdf:
                     results.append(stored_texts[idx])
                     seen_indices.add(idx)

        if not results:
             st.info(f"No specific context found for '{query}' in {selected_pdf}. Asking GPT-4 without specific context.")
             return "" # Return empty context if no relevant chunks found

        context = "\n\n---\n\n".join(results) # Use a clear separator
        return context[:4000]  # limit context size for GPT-4
    except Exception as e:
        st.error(f"Error during context retrieval: {e}")
        return ""


def ask_gpt4(query, context=""):
    """Ask GPT-4 using the context."""
    system_message = "You are a helpful assistant. Use the given context ONLY if it is provided and relevant to answer the question. If no context is provided, answer the question based on your general knowledge."
    if context and context.strip():
        user_content = f"Context:\n{context}\n\nQuestion: {query}"
    else:
        user_content = f"Question: {query}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content}
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.2,
            max_tokens=800
        )
        return response.choices[0].message['content']
    except Exception as e:
        st.error(f"Error calling GPT-4: {e}")
        return "Sorry, I encountered an error trying to generate a response."

def extract_text_from_pdf(uploaded_pdf):
    """Extract and clean text from uploaded PDF."""
    texts = []
    try:
        reader = PdfReader(uploaded_pdf)
        for page in reader.pages:
            text = page.extract_text()
            if text and text.strip():
                # Basic cleaning (optional, can be expanded)
                cleaned_text = ' '.join(text.split())
                texts.append(cleaned_text)
    except Exception as e:
        st.error(f"Error reading PDF file {uploaded_pdf.name}: {e}")
    return texts

# --- New Function to Extract Text from CSV ---
def extract_text_from_csv(uploaded_file):
    """Extract text from uploaded CSV, treating each row as a document chunk."""
    texts = []
    file_name = uploaded_file.name
    # --- DEBUG ---
    st.write(f"DEBUG: Starting extract_text_from_csv for {file_name}")
    # --- END DEBUG ---
    try:
        # Decode the file content, trying common encodings
        content = None
        try:
            # --- DEBUG ---
            st.write("DEBUG: Attempting UTF-8 decoding...")
            # --- END DEBUG ---
            content = uploaded_file.getvalue().decode('utf-8')
            # --- DEBUG ---
            st.write("DEBUG: UTF-8 decoding successful.")
            # --- END DEBUG ---
        except UnicodeDecodeError:
            st.warning(f"UTF-8 decoding failed for {file_name}, trying latin-1.")
            uploaded_file.seek(0) # Reset file pointer
            # --- DEBUG ---
            st.write("DEBUG: Attempting latin-1 decoding...")
            # --- END DEBUG ---
            content = uploaded_file.getvalue().decode('latin-1')
            # --- DEBUG ---
            st.write("DEBUG: latin-1 decoding successful.")
            # --- END DEBUG ---

        # Use io.StringIO to treat the string content as a file for the csv reader
        csvfile = io.StringIO(content)
        reader = None # Initialize reader
        # Sniff the dialect to handle different CSV formats (commas, tabs, etc.)
        try:
             # --- DEBUG ---
             st.write("DEBUG: Attempting to sniff CSV dialect...")
             # --- END DEBUG ---
             dialect = csv.Sniffer().sniff(csvfile.read(1024))
             csvfile.seek(0) # Reset after sniffing
             reader = csv.reader(csvfile, dialect)
             # --- DEBUG ---
             st.write(f"DEBUG: CSV dialect sniffed successfully: Delimiter='{dialect.delimiter}', Quotechar='{dialect.quotechar}'")
             # --- END DEBUG ---
        except csv.Error:
             st.warning(f"Could not automatically detect CSV dialect for {file_name}. Assuming comma-separated.")
             csvfile.seek(0) # Reset file pointer
             reader = csv.reader(csvfile) # Default to comma delimiter
             # --- DEBUG ---
             st.write("DEBUG: CSV dialect sniffing failed, assuming comma delimiter.")
             # --- END DEBUG ---

        header = next(reader, None) # Read header row
        header_text = ", ".join(header).strip() if header else ""
        # --- DEBUG ---
        st.write(f"DEBUG: CSV Header: {header_text[:100]}...") # Log first 100 chars of header
        # --- END DEBUG ---

        row_count = 0
        for i, row in enumerate(reader):
            # --- DEBUG ---
            if i % 100 == 0: # Log progress every 100 rows
                 st.write(f"DEBUG: Processing CSV row {i+1}...")
            # --- END DEBUG ---
            # Join non-empty cells in the row
            row_values = [cell.strip() for cell in row if cell and cell.strip()]
            if not row_values:
                continue # Skip rows that are entirely empty or whitespace

            row_text = ", ".join(row_values)

            # Combine header with row for context, clearly indicating row number
            # Limit length to avoid overly long chunks
            full_row_context = f"Row {i+1}: {row_text}"
            if header_text:
                 full_row_context = f"Header: {header_text}\n{full_row_context}"

            texts.append(full_row_context[:2000]) # Add chunk, limit length
            row_count += 1

        # --- DEBUG ---
        st.write(f"DEBUG: Finished processing CSV. Found {row_count} data rows. Total text chunks created: {len(texts)}")
        # --- END DEBUG ---
        if row_count == 0:
             st.warning(f"No data rows found in CSV file {file_name}.")

    except Exception as e:
        st.error(f"Error processing CSV file {file_name}: {e}")
        # --- DEBUG ---
        st.write(f"DEBUG: Error occurred in extract_text_from_csv: {e}")
        # --- END DEBUG ---
        return [] # Return empty list on error
    return texts
# --- End of New Function ---


def summarize_text(full_text):
    """Summarize the entire document."""
    # --- DEBUG ---
    st.write(f"DEBUG: Starting summarize_text. Input text length: {len(full_text)}")
    # --- END DEBUG ---
    if not full_text or not full_text.strip():
        return "Document appears to be empty or contains no text."
    # Limit the text sent for summarization to avoid excessive token usage/cost
    prompt = f"Please provide a concise summary of the following document content:\n\n{full_text[:10000]}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", # Use a cheaper model for summarization if acceptable
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500 # Reduced max tokens for summary
        )
        # --- DEBUG ---
        st.write("DEBUG: Summarization API call successful.")
        # --- END DEBUG ---
        return response.choices[0].message['content']
    except Exception as e:
        st.error(f"Error calling OpenAI for summarization: {e}")
        # --- DEBUG ---
        st.write(f"DEBUG: Error during summarization: {e}")
        # --- END DEBUG ---
        return "Could not generate summary due to an error."

# ------------------------------
# STREAMLIT UI
# ------------------------------
st.set_page_config(page_title="💬 Chat with Docs + GPT-4", page_icon="📄")
st.title("📄 Chat with your PDF & CSV Files + GPT-4")

# Session states
if "pdf_files" not in st.session_state:
    st.session_state.pdf_files = {} # Renaming to 'processed_files' might be clearer
    # Consider renaming pdf_files to processed_files or similar if handling multiple types

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Upload Section
with st.sidebar:
    # --- Updated File Uploader ---
    uploaded_files = st.file_uploader(
        "Upload PDF or CSV files",
        type=["pdf", "csv"], # Accept both pdf and csv
        accept_multiple_files=True
    )
    # --- End of Update ---

    if uploaded_files:
        processed_count = 0
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            # --- DEBUG ---
            st.write(f"DEBUG: Processing uploaded file: {file_name}")
            # --- END DEBUG ---
            # Check if file already processed (simple check by name)
            if file_name in st.session_state.pdf_files:
                 st.warning(f"File '{file_name}' already processed. Skipping.")
                 # --- DEBUG ---
                 st.write(f"DEBUG: Skipping already processed file: {file_name}")
                 # --- END DEBUG ---
                 continue

            texts = []
            # --- Updated Extraction Logic ---
            if file_name.lower().endswith('.pdf'):
                # --- DEBUG ---
                st.write(f"DEBUG: Calling extract_text_from_pdf for {file_name}")
                # --- END DEBUG ---
                texts = extract_text_from_pdf(uploaded_file)
            elif file_name.lower().endswith('.csv'):
                # --- DEBUG ---
                st.write(f"DEBUG: Calling extract_text_from_csv for {file_name}")
                # --- END DEBUG ---
                texts = extract_text_from_csv(uploaded_file)
            else:
                st.warning(f"Unsupported file type: {file_name}. Skipping.")
                # --- DEBUG ---
                st.write(f"DEBUG: Skipping unsupported file type: {file_name}")
                # --- END DEBUG ---
                continue
            # --- End of Update ---

            # --- DEBUG ---
            st.write(f"DEBUG: Text extraction completed for {file_name}. Number of text chunks: {len(texts)}")
            # --- END DEBUG ---

            if texts:
                # Embed and store the extracted text chunks
                # --- DEBUG ---
                st.write(f"DEBUG: Calling embed_and_store_text for {file_name}")
                # --- END DEBUG ---
                embed_and_store_text(texts, file_name)
                # --- DEBUG ---
                st.write(f"DEBUG: Finished embed_and_store_text for {file_name}")
                # --- END DEBUG ---

                # Generate summary (consider doing this lazily later if slow)
                full_text = "\n".join(texts) # Join chunks for summary context
                # --- DEBUG ---
                st.write(f"DEBUG: Calling summarize_text for {file_name}. Full text length: {len(full_text)}")
                # --- END DEBUG ---
                summary = summarize_text(full_text)
                # --- DEBUG ---
                st.write(f"DEBUG: Finished summarize_text for {file_name}")
                # --- END DEBUG ---

                # Store file info in session state
                st.session_state.pdf_files[file_name] = {
                    "summary": summary,
                    "full_text": full_text # Storing full text might consume a lot of memory
                    # Consider storing only summary or removing full_text if memory becomes an issue
                }
                processed_count += 1
                # --- DEBUG ---
                st.write(f"DEBUG: Stored info for {file_name} in session state.")
                # --- END DEBUG ---
            else:
                st.error(f"Could not extract text from {file_name}.")
                # --- DEBUG ---
                st.write(f"DEBUG: No text extracted from {file_name}, skipping embedding/summarization.")
                # --- END DEBUG ---

        if processed_count > 0:
             st.success(f"{processed_count} new file(s) processed and indexed!")
        st.info(f"Total indexed files: {len(st.session_state.pdf_files)}")
        st.write(f"Current FAISS index size: {index.ntotal} vectors.") # Debugging info


# Select which File to chat with
selected_file = None # Renamed from selected_pdf for clarity
if st.session_state.pdf_files:
    # Sort file list for consistent order
    file_options = sorted(list(st.session_state.pdf_files.keys()))
    selected_file = st.selectbox(
        "Select a file to chat with:",
        file_options
    )

    if selected_file:
        # Show Summary of Selected File
        st.subheader(f"📝 Summary for {selected_file}")
        st.markdown(st.session_state.pdf_files[selected_file]["summary"])
    else:
        st.info("Select a file from the dropdown above to start chatting.")


# Display previous chat messages
for msg in st.session_state.messages:
    # Simple way to handle potential missing keys
    role = msg.get("role", "unknown")
    content = msg.get("content", "[message content missing]")
    with st.chat_message(role):
        st.markdown(content)


# Chat Input
query = st.chat_input("Ask a question about your selected document...")
if query and selected_file:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Process and get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Retrieve context specific to the selected file
            context = retrieve_relevant_context(query, selected_file)
            # Get answer from GPT-4, potentially using the context
            answer = ask_gpt4(query, context)
            st.markdown(answer)

            # Show context only if it was actually used and not empty
            if context and context.strip():
                with st.expander("📚 Sources (Context Used)"):
                    st.markdown(context.replace("\n", "  \n")) # Ensure markdown newlines render

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})

elif query and not selected_file:
    st.warning("Please select a file from the dropdown before asking a question.")
