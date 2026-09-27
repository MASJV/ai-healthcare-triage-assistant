"""
knowledge_agent.py

Stage: Question -> Hybrid Retrieval (Vector + BM25, fused with RRF)
       -> BGE Reranker -> Top Chunks -> LLM (config.CHAT_MODEL) -> Answer with sources

Reads the Chroma collection built by ingest.py. Run `python ingest.py` first.
"""

import re

import chromadb
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

import config
from state import UserState


# --- Load once when the graph starts ---
openai_client = OpenAI()

chroma_client = chromadb.PersistentClient(path=config.CHROMA_DIRECTORY)
try:
    collection = chroma_client.get_collection(config.COLLECTION_NAME)
except Exception:
    raise RuntimeError("Knowledge base not found. Run `python ingest.py` first.")

# all chunks, kept in memory for BM25 keyword search
stored = collection.get(include=["documents", "metadatas"])
all_chunks = []
for i in range(len(stored["ids"])):
    all_chunks.append({
        "id": stored["ids"][i],
        "text": stored["documents"][i],
        "title": stored["metadatas"][i]["title"],
        "source": stored["metadatas"][i]["source"],
        "url": stored["metadatas"][i]["url"]
    })

# downloads the model once and keeps it in memory
reranker_model = CrossEncoder(config.RERANKER_MODEL)


def tokenize(text):
    """
    Turns text into a list of lowercase words for BM25.
    The regex keeps only letters/numbers, so "fever," and "fever" become the same word.
    Example: "Sore throat, fever?" -> ["sore", "throat", "fever"]
    """
    return re.findall(r"\w+", text.lower())


def build_bm25_index(chunks):
    """
    Builds a BM25 keyword search index from all the chunks.
    """
    tokenized_texts = []
    for chunk_info in chunks:
        tokenized_texts.append(tokenize(chunk_info["text"]))

    return BM25Okapi(tokenized_texts)


bm25_index = build_bm25_index(all_chunks)


def get_score(pair):
    """
    Tells sort() to order (score, chunk) pairs by their score.
    """
    return pair[0]


def keep_top_chunks(scored_chunks, top_k):
    """
    Takes a list of (score, chunk) pairs, sorts it with the highest
    score first and returns only the best top_k chunks.
    """
    scored_chunks.sort(key=get_score, reverse=True)

    top_chunks = []
    for score, chunk_info in scored_chunks[:top_k]:
        top_chunks.append(chunk_info)

    return top_chunks


def vector_search(question, top_k):
    """
    Finds the top_k chunks whose meaning is closest to the question.
    1. Turn the question into an embedding (a list of numbers) with OpenAI.
    2. Ask Chroma for the chunks whose embeddings are closest to it.
    """
    response = openai_client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=question,
        dimensions=config.EMBEDDING_DIMENSIONS
    )
    results = collection.query(
        query_embeddings=[response.data[0].embedding],
        n_results=top_k
    )

    matched_chunks = []
    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]
        matched_chunks.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "title": metadata["title"],
            "source": metadata["source"],
            "url": metadata["url"]
        })

    return matched_chunks


def bm25_search(question, top_k):
    """
    Finds the top_k chunks that best match the question's keywords.
    """
    scores = bm25_index.get_scores(tokenize(question))

    scored_chunks = []
    for i in range(len(all_chunks)):
        scored_chunks.append((scores[i], all_chunks[i]))

    return keep_top_chunks(scored_chunks, top_k)


def fuse_results(vector_chunks, bm25_chunks):
    """
    Combines the vector list and the BM25 list into one list using
    Reciprocal Rank Fusion (RRF).

    Each chunk gets 1 / (60 + rank) from every list it appears in.
    A chunk near the top of either list scores high, a chunk near
    the top of BOTH lists scores higher.
    """
    combined_scores = {}
    all_chunks_by_id = {}

    for result_list in [vector_chunks, bm25_chunks]:
        for rank, chunk_info in enumerate(result_list):
            chunk_id = chunk_info["id"]
            score = 1 / (config.RRF_K + rank + 1)

            if chunk_id not in combined_scores:
                combined_scores[chunk_id] = 0
            combined_scores[chunk_id] = combined_scores[chunk_id] + score
            all_chunks_by_id[chunk_id] = chunk_info

    scored_chunks = []
    for chunk_id in combined_scores:
        scored_chunks.append((combined_scores[chunk_id], all_chunks_by_id[chunk_id]))

    # keep every chunk, just in the new fused order
    return keep_top_chunks(scored_chunks, len(scored_chunks))


def rerank_chunks(question, chunks, top_k):
    """
    Re-scores every chunk against the question with the BGE cross-encoder
    and keeps only the best top_k.
    """
    pairs = []
    for chunk_info in chunks:
        pairs.append([question, chunk_info["text"]])

    scores = reranker_model.predict(pairs)

    scored_chunks = []
    for i in range(len(chunks)):
        scored_chunks.append((scores[i], chunks[i]))

    return keep_top_chunks(scored_chunks, top_k)


def retrieve(question):
    """
    Full retrieval pipeline: hybrid search -> fusion -> rerank.
    """
    vector_chunks = vector_search(question, config.TOP_K_VECTOR)
    bm25_chunks = bm25_search(question, config.TOP_K_BM25)
    fused_chunks = fuse_results(vector_chunks, bm25_chunks)
    return rerank_chunks(question, fused_chunks, config.TOP_K_FINAL)


def build_context_text(chunks):
    """
    Joins the chunks into one string, numbering each one so the
    answer can cite it as [1], [2], ...
    """
    context_parts = []
    for number, chunk_info in enumerate(chunks, start=1):
        context_parts.append(
            f"[{number}] {chunk_info['title']} ({chunk_info['source']})\n{chunk_info['text']}"
        )
    return "\n\n---\n\n".join(context_parts)


DISCLAIMER = (
    "Disclaimer: This is general self-care information, not a medical diagnosis. "
    "Please consult a doctor for medical advice."
)


#Knowledge Agent
def knowledge_agent(state: UserState):

    chunks = retrieve(state["user_input"])

    prompt = f"""
    You are a healthcare self-care information assistant.
    Answer the user's message using only the context below.

    Rules:
    - Do not give a diagnosis.
    - Give general self-care guidance.
    - End with one line on when to see a doctor
      (e.g. "See a doctor if the fever gets worse or lasts more than 3 days.").
    - Do not recommend prescription medicines (e.g. antibiotics) or give doses;
      point the user to a doctor or pharmacist instead.
    - Cite the context you used as [1], [2], etc.
    - If the context does not answer the message, say you do not have
      information on this and suggest consulting a doctor.
    - The context may mention UK numbers (999, 111). Use Indian ones instead:
      112 for emergencies, 108 for an ambulance.
    - Keep the answer within 8 lines.

    Intent: {state['intent']}

    Context:
    {build_context_text(chunks)}

    User Message: {state['user_input']}
    """

    response = config.llm.invoke(prompt)

    source_lines = []
    for number, chunk_info in enumerate(chunks, start=1):
        source_lines.append(f"[{number}] {chunk_info['title']} - {chunk_info['url']}")
    sources = "\n".join(source_lines)

    return {
        "result": response.content + "\n\n" + DISCLAIMER + "\n\nSources:\n" + sources
    }
