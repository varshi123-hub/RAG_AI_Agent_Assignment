"""
rag_agent.py - LangGraph RAG Agent

This module implements a simple RAG (Retrieval-Augmented Generation) agent
using LangGraph. The agent follows this flow:

    [User Query] -> [Retrieve Chunks from ChromaDB] -> [Generate Answer from Chunks] -> [Final Response]

The graph has two nodes:
    1. retrieve  - Queries the vector DB for relevant document chunks
    2. generate  - Sends the chunks + question to the LLM for a grounded answer
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# --- Configuration ---
CHROMA_DB_DIR = "chroma_db"  # Must match ingestion.py
TOP_K = 4  # Number of chunks to retrieve per query


# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------
# LangGraph agents pass a shared "state" dict between nodes.
# We define the shape of that state here using TypedDict.

class RAGState(TypedDict):
    question: str  # The user's question
    context: str  # Retrieved document chunks (joined text)
    answer: str  # The final generated answer


# ---------------------------------------------------------------------------
# Node 1: Retrieve relevant chunks from ChromaDB
# ---------------------------------------------------------------------------

def retrieve(state: RAGState) -> dict:
    """
    RAG Step 1 - Query the vector DB.

    Takes the user's question, performs a similarity search against
    ChromaDB, and returns the top-K most relevant chunks as context.
    """
    question = state["question"]

    # Load the persisted ChromaDB vector store
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
    )

    # Perform similarity search - finds chunks closest to the question
    results = vector_store.similarity_search(question, k=TOP_K)

    # Combine the chunk texts into a single context string
    context = "\n\n---\n\n".join([doc.page_content for doc in results])

    print(f"[Retrieve] Found {len(results)} relevant chunks for: '{question}'")
    return {"context": context}


# ---------------------------------------------------------------------------
# Node 2: Generate an answer using the LLM + retrieved context
# ---------------------------------------------------------------------------

def generate(state: RAGState) -> dict:
    """
    RAG Step 2 - Answer from RAG chunks.

    Takes the retrieved context and the original question, sends them
    to the LLM with a prompt that instructs it to answer based only
    on the provided context.
    """
    question = state["question"]
    context = state["context"]

    # Define the RAG prompt template
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant. Answer the user's question based ONLY "
            "on the following context from retrieved documents. If the context "
            "does not contain enough information, say so honestly.\n\n"
            "Context:\n{context}"
        ),
        ("human", "{question}"),
    ])

    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Build the chain: prompt -> LLM
    chain = prompt | llm

    # Invoke the chain with our context and question
    response = chain.invoke({"context": context, "question": question})

    print(f"[Generate] Answer produced.")
    return {"answer": response.content}


# ---------------------------------------------------------------------------
# Build the LangGraph Agent
# ---------------------------------------------------------------------------

def build_rag_agent() -> StateGraph:
    """
    Constructs the LangGraph RAG agent with two nodes:

        START -> retrieve -> generate -> END

    This is intentionally kept simple to clearly show the RAG flow:
    1. User asks a question
    2. retrieve node fetches relevant chunks from ChromaDB
    3. generate node uses those chunks to produce a grounded answer
    4. The answer is returned
    """
    # Create the state graph with our RAGState schema
    graph = StateGraph(RAGState)

    # Add the two nodes
    graph.add_node("retrieve", retrieve)  # Node 1: Query vector DB
    graph.add_node("generate", generate)  # Node 2: Generate answer

    # Define the edges (flow): START -> retrieve -> generate -> END
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    # Compile the graph into a runnable agent
    agent = graph.compile()
    return agent


def query_rag(question: str) -> str:
    """
    Convenience function to run a single question through the RAG agent.

    Args:
        question: The user's question string.

    Returns:
        The agent's answer as a string.
    """
    agent = build_rag_agent()

    # Run the agent with the initial state
    result = agent.invoke({
        "question": question,
        "context": "",  # Will be filled by the retrieve node
        "answer": "",  # Will be filled by the generate node
    })

    return result["answer"]


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    # Quick test
    answer = query_rag("What is this document about?")
    print(f"\nAnswer: {answer}")
