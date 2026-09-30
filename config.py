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
CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536  # full size of text-embedding-3-small
JUDGE_MODEL = "gpt-4.1-mini"  # grades the knowledge agent in eval_knowledge.py

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

# --- Triage settings ---
MAX_FOLLOW_UPS = 2  # follow-up questions triage may ask before it must decide

# --- Scheduling settings ---
DB_PATH = "appointments.db"
BOOKING_DAYS = 30    # users can book up to 30 days ahead
URGENT_HOURS = 48    # urgent cases must be booked within 48 hours
OPEN_HOUR = 8        # first slot at 8:00 AM
CLOSE_HOUR = 22      # last slot ends at 10:00 PM
SLOT_MINUTES = 30    # length of one appointment
SHOW_HOURS = 24      # free slots of the next 24 hours are shown to the user

# --- Storage settings ---
DOCS_FOLDER = "health_docs_clean/health_docs_clean"
CHROMA_DIRECTORY = "chroma_db"
COLLECTION_NAME = "health_docs"

# --- Shared LLM used by all agents ---
llm = ChatOpenAI(
    model=CHAT_MODEL,
    temperature=0
)
