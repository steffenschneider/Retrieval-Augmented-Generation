# Retrieval-Augmented-Generation

- a new project for the next weeks :)
- difficult but very important and interesting

Technology Stack
Component	Tool / Library
- Document Loading	LangChain, PyMuPDF, PDFMiner, Docx2txt, etc.
- Text Splitting	LangChain.TextSplitter
- Embeddings	sentence-transformers, OpenAI Embeddings, or HuggingFace
- Vector Store	FAISS (lightweight, local), or ChromaDB
- Retriever	LangChain Retriever or custom FAISS similarity search
- Language Model (LLM)	OpenAI GPT, HuggingFace Transformers, LLaMA
- Optional UI	Streamlit, Gradio, or simple CLI

Workflow
- Load documents
    - Read local files (.txt, .pdf, .docx, etc.)
    - Clean and preprocess the content
- Split text into chunks
    - Typically 300–1000 tokens each
    - Use overlap (e.g. 50–100 tokens) for better context retention
- Generate vector embeddings
    - Convert each chunk into an embedding (vector representation)
- Store in vector database (FAISS)
    - Allows fast semantic similarity search
- User asks a question
    - Input gets embedded and matched with top-k most similar chunks
- Retrieve top context chunks
    - Use those chunks to provide context for the LLM
- LLM generates an answer
    - Based on retrieved information and prompt template
 
Run LLM locally? 
- "High-quality" answers need high-quality language models, which usually don't run well locally — especially not with just 8 GB RAM.
- What people do instead (including pros):
    - Local embeddings → run locally, fast, accurate for finding relevant chunks
    - Cloud-based LLM → only the answering happens online
    - This hybrid setup is called a "local+cloud RAG" and is what most devs and startups use.

Model choice:
- fast + smart --> OpenAI GPT-3.5    (cost 0,002$ / request --> affordable)


