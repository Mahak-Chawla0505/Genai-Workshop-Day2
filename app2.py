import streamlit as st
from dotenv import load_dotenv

# Langchain imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="C++ RAG Chatbot",
    layout="wide"
)

st.title("💭 C++ RAG Chatbot")
st.write("Powered by Gemma2 (Ollama)")

load_dotenv()

# -----------------------
# LOAD VECTOR STORE
# -----------------------
@st.cache_resource
def load_vector_store():

    # Load text file
    loader = TextLoader("C++_Introduction.txt", encoding="utf-8")
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create FAISS database
    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    return vectorstore


vectorstore = load_vector_store()

# -----------------------
# LOAD GEMMA MODEL
# -----------------------
llm = Ollama(
    model="gemma2:2b"
)

# -----------------------
# PROMPT TEMPLATE
# -----------------------
prompt = ChatPromptTemplate.from_template("""
Answer the question based ONLY on the context below.

Context:
{context}

Question:
{question}
""")

# -----------------------
# USER INPUT
# -----------------------
question = st.text_input("Ask a question about C++")

# -----------------------
# RUN MODEL
# -----------------------
if question:

    # Animation while thinking
    with st.spinner("Gemma is thinking... 🤔"):

        # Search similar chunks
        docs = vectorstore.similarity_search(
            question
        )

        # Combine context
        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        # Create prompt
        final_prompt = prompt.format(
            context=context,
            question=question
        )

        # Get response
        response = llm.invoke(final_prompt)

    # Show results
    st.subheader("Answer")
    st.write(response)

    st.subheader("Retrieved Context")
    for i, doc in enumerate(docs):
        st.write(f"Chunk {i+1}")
        st.write(doc.page_content)