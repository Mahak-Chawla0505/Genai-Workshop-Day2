# Import required libraries
import os
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
# Loads text documents
from langchain_community.document_loaders import TextLoader

# Splits long text into smaller chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Converts text into embeddings (numbers AI understands)
from langchain_huggingface import HuggingFaceEmbeddings

# Vector database for storing embeddings
from langchain_community.vectorstores import FAISS


# -------------------------------
# Step 1: Page Configuration
# -------------------------------

# Sets browser tab title and icon
st.set_page_config(page_title="C++ RAG Chatbot", page_icon="💭")

# App title
st.title("💭 C++ RAG Chatbot")

# Description text
st.write("Feel free to ask anything about C++")


# -------------------------------
# Step 2: Load Environment Variables
# -------------------------------

# Loads variables from .env file (if any)
# Example: API keys
load_dotenv()


# -------------------------------
# Step 3: Create Vector Store
# -------------------------------
# We use caching so this runs only once
# Otherwise embeddings would regenerate every time
@st.cache_resource
def load_vector_store():

    # -------------------------------
    # Step A: Load Documents
    # -------------------------------

    # Load text file containing C++ information
    loader = TextLoader("C++_Introduction.txt", encoding="utf-8")

    # Convert text into document format
    documents = loader.load()


    # -------------------------------
    # Step B: Split Text into Chunks
    # -------------------------------

    # Splits large text into smaller chunks
    # chunk_size = characters per chunk
    # chunk_overlap = overlapping characters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
        # Overlap helps maintain context between chunks
    )

    # Apply splitting
    final_documents = text_splitter.split_documents(documents)


    # -------------------------------
    # Step C: Create Embeddings
    # -------------------------------

    # Embedding model converts text -> numbers
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        # Lightweight and fast embedding model
    )


    # -------------------------------
    # Step D: Create FAISS Vector Store
    # -------------------------------

    # FAISS stores embeddings and allows searching
    vector_store = FAISS.from_documents(
        final_documents,
        embeddings
    )

    # Return vector database
    return vector_store


# -------------------------------
# Step 4: Load Vector Store
# -------------------------------

# Runs load_vector_store() once
vector_store = load_vector_store()


# -------------------------------
# Step 5: User Input
# -------------------------------

# Text input box
user_input = st.text_input("Enter your question about C++:")


# -------------------------------
# Step 6: Search Similar Documents
# -------------------------------

if user_input:

    # Convert question into embeddings
    # Search FAISS database
    # Retrieve top 3 similar chunks
    documents = vector_store.similarity_search(
        user_input,
        k=3
    )

    # Display section title
    st.subheader("🟨 Retrieved Context")

    # Show results
    for i, doc in enumerate(documents):

        # Result number
        st.markdown(f"**Result {i+1}:**")

        # Show text content
        st.write(doc.page_content)