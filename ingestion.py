"""
ingestion.py - PDF Data Extraction & ChromaDB Vector Store Creation

This module handles the first phase of the RAG pipeline:
1. Load PDF documents from the 'data/' folder
2. Split them into smaller chunks for better retrieval
3. Create embeddings and store them in a local ChromaDB vector database
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
DATA_DIR = "data"  # Folder where PDF files are placed
CHROMA_DB_DIR = "chroma_db"  # Folder where ChromaDB will persist vectors
CHUNK_SIZE = 1000  # Number of characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks to preserve context


def load_pdfs(data_dir: str) -> list:
    """
    Step 1: Data Extraction - Load all PDF files from the data directory.

    Reads every .pdf file in the given folder and returns a flat list
    of LangChain Document objects (one per page).
    """
    documents = []

    # Check if data directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created '{data_dir}/' folder. Please add PDF files there and re-run.")
        return documents

    # Loop through all PDF files in the directory
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in '{data_dir}/'. Please add some and re-run.")
        return documents

    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_dir, pdf_file)
        print(f"Loading: {pdf_file}")

        # PyPDFLoader extracts text from each page of the PDF
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

    print(f"Loaded {len(documents)} pages from {len(pdf_files)} PDF(s).")
    return documents


def split_documents(documents: list) -> list:
    """
    Step 2: Chunking - Split documents into smaller overlapping chunks.

    Large pages are broken into smaller pieces so the retriever can
    return only the most relevant portion instead of an entire page.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


def create_vector_store(chunks: list) -> Chroma:
    """
    Step 3: Create Vector DB - Embed chunks and store in ChromaDB.

    Each chunk is converted into a vector embedding using OpenAI's
    embedding model, then stored locally in ChromaDB for fast
    similarity search during retrieval.
    """
    # Initialize the embedding model
    embeddings = OpenAIEmbeddings()

    # Create (or overwrite) the ChromaDB vector store on disk
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )

    print(f"Vector store created at '{CHROMA_DB_DIR}/' with {len(chunks)} vectors.")
    return vector_store


def run_ingestion() -> Chroma | None:
    """
    Main ingestion pipeline: Extract -> Chunk -> Vectorize.

    Returns the ChromaDB vector store if successful, None otherwise.
    """
    print("=" * 50)
    print("STEP 1: DATA EXTRACTION (PDF Loading)")
    print("=" * 50)
    documents = load_pdfs(DATA_DIR)
    if not documents:
        return None

    print()
    print("=" * 50)
    print("STEP 2: CHUNKING (Text Splitting)")
    print("=" * 50)
    chunks = split_documents(documents)

    print()
    print("=" * 50)
    print("STEP 3: CREATE VECTOR DB (ChromaDB)")
    print("=" * 50)
    vector_store = create_vector_store(chunks)

    print("\nIngestion complete!")
    return vector_store


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_ingestion()
