#!pip install langchain
#!pip install langchain-community
#!pip install faiss-cpu
#!pip install openai
#!pip install sentence-transformers
#!pip install tf-keras

import os
import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

vector_store_path = "C:/Users/Steffen/Desktop/vector_store_1"
openai_key_path = "C:/Users/Steffen/Desktop/openai_key.txt"

# Load OpenAI API key from file
def load_api_key():
    try:
        with open(openai_key_path, "r") as f:
            key = f.read().strip()
            os.environ["OPENAI_API_KEY"] = key
            return True
    except Exception as e:
        st.error(f"Error loading API key: {str(e)}")
        return False

# 1. Load .txt files from 'documents' folder
def load_documents(folder_path="C:/Users/Steffen/Dropbox"):
    docs = []
    for file in os.listdir(folder_path):
        if file.startswith("lex-home") and file.endswith(".txt"):
            path = os.path.join(folder_path, file)
            loader = TextLoader(path, encoding='utf-8')
            docs.extend(loader.load())
    return docs

# 2. Split documents into chunks
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=240, chunk_overlap=30)
    return splitter.split_documents(documents)

# 3. Create a RAG chain using OpenAI GPT
def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo")
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Streamlit app
st.title("Document Q&A System")

# Initialize session state variables if they don't exist
if 'qa_ready' not in st.session_state:
    st.session_state.qa_ready = False
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'api_key_loaded' not in st.session_state:
    st.session_state.api_key_loaded = load_api_key()

# Sidebar for configuration and loading
with st.sidebar:
    st.header("Configuration")
    
    # Show API key status
    if st.session_state.api_key_loaded:
        st.success("API key loaded successfully!")
    else:
        st.error("API key not loaded. Check the file path.")
    
    # Option to use local embeddings
    use_local = st.radio("Use existing vector store?", ["Yes", "No"], index=0)
    
    # Load button
    if st.session_state.api_key_loaded and st.button("Initialize System"):
        with st.spinner("Setting up the system..."):
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            if use_local == "Yes" and os.path.exists(vector_store_path):
                st.info("Loading existing vector store...")
                vectorstore = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
                st.success("Vector store loaded successfully!")
            else:
                st.info("Loading documents...")
                docs = load_documents()
                st.info(f"Loaded {len(docs)} documents.")
                
                st.info("Splitting into chunks...")
                chunks = split_documents(docs)
                st.info(f"{len(chunks)} chunks created.")
                
                st.info("Creating new vector store...")
                vectorstore = FAISS.from_documents(chunks, embeddings)
                vectorstore.save_local(vector_store_path)
                st.success("Vector store created and saved!")
            
            st.info("Initializing question-answering chain...")
            st.session_state.qa_chain = create_qa_chain(vectorstore)
            st.session_state.qa_ready = True
            st.success("System is ready for questions!")

# Main area for Q&A
if st.session_state.qa_ready:
    st.header("Ask a Question")
    query = st.text_input("Your question about the documents:", key="question_input")
    
    if st.button("Submit Question"):
        if query:
            with st.spinner("Finding answer..."):
                answer = st.session_state.qa_chain.run(query)
                
            st.subheader("Answer:")
            st.write(answer)
        else:
            st.warning("Please enter a question.")
else:
    st.info("Please initialize the system using the sidebar controls.")
    
    # Optional instructions
    with st.expander("How to use this app"):
        st.write("""
        1. The OpenAI API key is loaded automatically from your file
        2. Choose whether to use an existing vector store or create a new one
        3. Click 'Initialize System' to set up
        4. Once initialized, ask questions about your documents in the main panel
        """)