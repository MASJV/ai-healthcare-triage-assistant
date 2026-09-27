"""
config.py

This file keeps all the settings in one place.
If you want to change a model name or a number later,
you only need to change it here, not in every file.
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load OPENAI_API_KEY from .env
load_dotenv(override=True)

# --- Models ---
CHAT_MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 512   # smaller vectors keep local storage small

# --- Free local reranker model (no API cost) ---
RERANKER_MODEL = "BAAI/bge-reranker-base"

# --- Chunking settings ---
CHUNK_SIZE = 2000      # characters (~500 tokens)
CHUNK_OVERLAP = 250    # characters (~12% overlap)

# --- Retrieval settings ---
TOP_K_VECTOR = 10   # how many chunks to get from vector search
TOP_K_BM25 = 10     # how many chunks to get from keyword search
TOP_K_FINAL = 4     # how many chunks to keep after reranking
RRF_K = 60          # standard Reciprocal Rank Fusion constant

# --- Storage settings ---
DOCS_FOLDER = "health_docs_clean/health_docs_clean"
CHROMA_DIRECTORY = "chroma_db"
COLLECTION_NAME = "health_docs"

# --- Shared LLM used by all agents ---
llm = ChatOpenAI(
    model=CHAT_MODEL,
    temperature=0
)
