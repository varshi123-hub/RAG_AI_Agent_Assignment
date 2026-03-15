# RAG AI Agent

A simple Retrieval-Augmented Generation (RAG) agent built with **LangGraph** and **ChromaDB**.

## Architecture

```
PDF Files ──► Ingestion Pipeline ──► ChromaDB Vector Store
                                            │
User Question ──► LangGraph RAG Agent ──────┤
                       │                    │
                   [Retrieve] ◄─────────────┘
                       │
                   [Generate] ──► Final Answer
```

### Phase 1: Ingestion (`ingestion.py`)
1. **Data Extraction** - Load PDF files from `data/` using PyPDF
2. **Chunking** - Split documents into overlapping chunks (1000 chars, 200 overlap)
3. **Vector DB Creation** - Embed chunks with OpenAI embeddings and store in ChromaDB

### Phase 2: RAG Agent (`rag_agent.py`)
A LangGraph state graph with two nodes:
1. **Retrieve** - Similarity search against ChromaDB to find relevant chunks
2. **Generate** - LLM answers the question using only the retrieved context

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create .env file with your OpenAI API key
cp .env.example .env
# Edit .env and add your key

# 3. Add PDF files to the data/ folder
mkdir data
# Copy your PDFs into data/

# 4. Run the agent
python main.py
```

## Project Structure

```
├── main.py           # Entry point - runs ingestion then interactive Q&A
├── ingestion.py      # PDF loading, chunking, ChromaDB vector store creation
├── rag_agent.py      # LangGraph RAG agent (retrieve + generate nodes)
├── requirements.txt  # Python dependencies
├── .env.example      # Template for environment variables
└── data/             # Place PDF files here (gitignored)
```

## How It Works

1. Place your PDF documents in the `data/` folder
2. Run `python main.py`
3. The system ingests PDFs into ChromaDB (first run only)
4. Ask questions interactively - the agent retrieves relevant chunks and generates answers
