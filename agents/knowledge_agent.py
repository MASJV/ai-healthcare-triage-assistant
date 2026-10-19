"""
knowledge_agent.py

Stage: Question -> Query Expansion -> Hybrid Retrieval (Vector + BM25, fused with RRF)
       -> BGE Reranker -> Top Chunks -> LLM (config.CHAT_MODEL) -> Answer with sources

Reads the Chroma collection built by ingest.py. Run `python ingest.py` first.
"""

import re

import chromadb
from langchain_core.messages import AIMessage
from langsmith import traceable
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

import config
from state import UserState


openai_client = OpenAI()

chroma_client = chromadb.PersistentClient(path=config.CHROMA_DIRECTORY)
try:
    collection = chroma_client.get_collection(config.COLLECTION_NAME)
except Exception:
    raise RuntimeError("Knowledge base not found. Run `python ingest.py` first.")

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


@traceable(name="vectorSearch")
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


@traceable(name="bm25Search")
def bm25_search(question, top_k):
    """
    Finds the top_k chunks that best match the question's keywords.
    """
    scores = bm25_index.get_scores(tokenize(question))

    scored_chunks = []
    for i in range(len(all_chunks)):
        scored_chunks.append((scores[i], all_chunks[i]))

    return keep_top_chunks(scored_chunks, top_k)


@traceable(name="fuseResults")
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


@traceable(name="rerankChunks")
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


@traceable(name="expandQuery")
def expand_query(question):
    """
    Adds medical terms to the question before searching, so everyday words
    still find the right documents (e.g. "piles" -> "hemorrhoids").
    The original question is kept, so nothing the user wrote is lost.
    If the LLM call fails, the original question is used alone.
    """
    prompt = f"""
    A user asked a health question. List up to 5 medical or standard English
    terms for the condition or topic they are asking about, separated by commas.
    Reply with the terms only.

    Question: {question}
    """

    try:
        response = config.llm.invoke(prompt)
    except Exception:
        return question

    return f"{question} {response.content.strip()}"


@traceable(name="retrieve")
def retrieve(question):
    """
    Full retrieval pipeline: query expansion -> hybrid search -> fusion -> rerank.
    """
    search_query = expand_query(question)
    vector_chunks = vector_search(search_query, config.TOP_K_VECTOR)
    bm25_chunks = bm25_search(search_query, config.TOP_K_BM25)
    fused_chunks = fuse_results(vector_chunks, bm25_chunks)
    return rerank_chunks(search_query, fused_chunks, config.TOP_K_FINAL)


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

# opening line when triage decided the user's problem can be managed at home
SELF_CARE_LINE = (
    "From what you have described, this can usually be managed at home for now. "
    "If it gets worse or new symptoms appear, tell me and I can help you book an appointment."
)

# a number (or ½, ¼, ¾) followed by a unit that is always a medicine dose, e.g. "2 tablets", "½ tablet"
MEDICINE_UNIT_PATTERN = r"(\d+(\.\d+)?|½|¼|¾)\s*(tablets?|capsules?|pills?|drops)\b"

# a number followed by a unit that is also used for food and drink, e.g. "500 mg", "5 ml";
# it only counts as a dose when the same sentence names a medicine
AMOUNT_PATTERN = r"(\d+(\.\d+)?|½|¼|¾)\s*(mg|mcg|ml|milligrams?|micrograms?|teaspoons?|tsp)\b"
MEDICINE_WORDS = (
    r"\b(medicine|medication|drug|dose|dosage|tablet|capsule|pill|syrup|supplement"
    r"|paracetamol|acetaminophen|ibuprofen|aspirin|antibiotic|antihistamine)"
)

NO_DOSE_LINE = "For the right dose of any medicine, please ask a doctor or pharmacist."


def has_dose(sentence):
    """
    True if the sentence gives a medicine dose.
    "Take 2 tablets" -> True, "75 mg of aspirin" -> True,
    "drink about 2,000 ml of water" -> False, "300 mg of cholesterol" -> False
    """
    text = sentence.lower()
    if re.search(MEDICINE_UNIT_PATTERN, text):
        return True
    if re.search(AMOUNT_PATTERN, text) and re.search(MEDICINE_WORDS, text):
        return True
    return False


@traceable(name="removeDoses")
def remove_doses(answer):
    """
    Safety check in code: the prompt says never give a dose, but the model
    sometimes still copies one from the documents.
    Drops every sentence that contains a dose and adds one line pointing
    the user to a doctor or pharmacist. Answers without a dose are unchanged.
    """
    kept_lines = []
    removed = False

    for line in answer.splitlines():
        kept_sentences = []
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            if has_dose(sentence):
                removed = True
            else:
                kept_sentences.append(sentence)
        kept_lines.append(" ".join(kept_sentences))

    if not removed:
        return answer

    kept_lines.append(NO_DOSE_LINE)
    return "\n".join(kept_lines).strip()


#Knowledge Agent
def knowledge_agent(state: UserState):

    # triage sent this here as self_care: search with all the user's messages
    # about the problem, not only the last answer to a follow-up question
    from_triage = state.get("urgency", "") == "self_care"
    if from_triage and state.get("symptom_text"):
        question = state["symptom_text"]
    else:
        question = state["user_input"]

    chunks = retrieve(question)

    prompt = f"""
    You are a healthcare self-care information assistant.
    Answer the user's message strictly from the context below.

    Safety rule (this overrides every other rule):
    - Do not recommend prescription medicines (e.g. antibiotics). Do not give a
      dose, amount or schedule for any medicine, including over-the-counter ones
      like paracetamol, even if the context has one; point the user to a doctor
      or pharmacist instead.

    Rules:
    - Use only facts stated in the context. Do not add advice, facts,
      numbers or home remedies from your own knowledge, even if you think
      they are correct.
    - Every sentence of guidance must be supported by the context and cited.
    - Only include information that answers what the user asked. Leave out
      facts from the context that do not answer it, even if they are on the
      same topic.
    - Do not give a diagnosis.
    - End with one line on when to see a doctor, taken from the context.
      If the context does not say, write: "See a doctor if your symptoms
      get worse or do not improve."
    - Cite the context you used as [1], [2], etc.
    - If the context does not answer the message, say you do not have
      information on this and suggest consulting a doctor.
      Do not say the context lacks information when it answers the question
      in different words or only briefly.
    - The context may mention UK numbers (999, 111). Use Indian ones instead:
      112 for emergencies, 108 for an ambulance.
    - Keep the answer within 8 lines.

    Intent: {state['intent']}

    Context:
    {build_context_text(chunks)}

    User Message: {question}
    """

    response = config.llm.invoke(prompt)
    answer_text = remove_doses(response.content)

    source_lines = []
    for number, chunk_info in enumerate(chunks, start=1):
        source_lines.append(f"[{number}] {chunk_info['title']} - {chunk_info['url']}")
    sources = "\n".join(source_lines)

    if from_triage:
        answer_text = f"{SELF_CARE_LINE}\n\n{answer_text}"

    answer = f"{answer_text}\n\n{DISCLAIMER}\n\nSources:\n{sources}"

    return {
        "result": answer,
        "messages": [AIMessage(content=answer)]
    }
