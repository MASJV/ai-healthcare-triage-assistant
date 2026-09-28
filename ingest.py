"""
ingest.py

Stage: Health Docs -> Markdown Parsing -> Chunking -> Embeddings -> Chroma

Run this once (and again whenever the documents change):
    python ingest.py

It reads every Markdown file in the knowledge base, splits it into
chunks, embeds the chunks and saves them in a Chroma collection on disk.
The Knowledge Agent then only reads from that collection.
"""

import os

import chromadb
from openai import OpenAI

import config


def load_documents(folder_path):
    """
    Reads every .md file under folder_path and returns a list of
    simple dictionaries:
        {"title": "...", "source": "...", "url": "...", "text": "..."}
    The title, source and url come from the front matter at the top of each file.
    """
    documents = []

    for root, _, file_names in os.walk(folder_path):
        for file_name in sorted(file_names):
            if not file_name.endswith(".md"):
                continue

            with open(os.path.join(root, file_name), encoding="utf-8") as input_file:
                content = input_file.read()

            metadata = {"title": file_name, "source": "unknown", "url": ""}
            body = content

            # front matter sits between the first two "---" lines
            if content.startswith("---"):
                parts = content.split("---", 2)
                for line in parts[1].strip().splitlines():
                    key, _, value = line.partition(":")
                    metadata[key.strip()] = value.strip().strip('"')
                body = parts[2]

            metadata["text"] = body.strip()
            documents.append(metadata)

    return documents


def chunk_text(text, chunk_size, overlap):
    """
    Plain sliding-window split, used only for sections that are
    too long to keep as one chunk.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks


def split_into_sections(text):
    """
    Splits a document on its "## " headings so that a section
    (e.g. "When to seek help") stays together where possible.
    """
    sections = []
    current_lines = []

    for line in text.splitlines():
        if line.startswith("## ") and current_lines:
            section = "\n".join(current_lines).strip()
            if section:
                sections.append(section)
            current_lines = []
        current_lines.append(line)

    section = "\n".join(current_lines).strip()
    if section:
        sections.append(section)

    return sections


def chunk_documents(documents, chunk_size, overlap):
    """
    Header-aware chunking:
    1. Split each document into sections on its headings.
    2. Join neighbouring sections while they fit inside chunk_size.
    3. A single section longer than chunk_size is split with a sliding window.

    Each chunk starts with the document title, so a chunk like
    "Treatment ..." still says which condition it belongs to.
    """
    all_chunks = []
    chunk_number = 0

    for document in documents:
        text_chunks = []
        current = ""

        for section in split_into_sections(document["text"]):
            if len(section) > chunk_size:
                if current:
                    text_chunks.append(current)
                    current = ""
                text_chunks.extend(chunk_text(section, chunk_size, overlap))
            elif len(current) + len(section) + 2 <= chunk_size:
                if current:
                    current = f"{current}\n\n{section}"
                else:
                    current = section
            else:
                text_chunks.append(current)
                current = section

        if current:
            text_chunks.append(current)

        for text_chunk in text_chunks:
            chunk_info = {
                "id": f"chunk_{chunk_number}",
                "text": f"{document['title']}\n\n{text_chunk}",
                "title": document["title"],
                "source": document["source"],
                "url": document["url"]
            }
            all_chunks.append(chunk_info)
            chunk_number = chunk_number + 1

    return all_chunks


def get_embeddings(texts, client):
    """
    Sends a batch of texts to OpenAI and gets back one embedding per text.
    """
    response = client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=texts,
        dimensions=config.EMBEDDING_DIMENSIONS
    )
    embeddings = []
    for item in response.data:
        embeddings.append(item.embedding)
    return embeddings


def build_vector_store(chunks, batch_size=100):
    """
    Creates a fresh Chroma collection and stores every chunk's
    embedding inside it.
    """
    client = OpenAI()
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_DIRECTORY)

    # remove old collection if it exists, so each run starts clean
    for existing_collection in chroma_client.list_collections():
        if existing_collection.name == config.COLLECTION_NAME:
            chroma_client.delete_collection(config.COLLECTION_NAME)

    collection = chroma_client.create_collection(
        config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # send 100 chunks per OpenAI call instead of one call per chunk
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]

        ids = []
        texts = []
        metadatas = []
        for chunk_info in batch:
            ids.append(chunk_info["id"])
            texts.append(chunk_info["text"])
            metadatas.append({
                "title": chunk_info["title"],
                "source": chunk_info["source"],
                "url": chunk_info["url"]
            })

        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=get_embeddings(texts, client)
        )
        print(f"Embedded {min(start + batch_size, len(chunks))}/{len(chunks)} chunks")

    return collection


if __name__ == "__main__":
    documents = load_documents(config.DOCS_FOLDER)
    print(f"Loaded {len(documents)} documents")

    chunks = chunk_documents(documents, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    print(f"Created {len(chunks)} chunks")

    build_vector_store(chunks)
    print(f"Saved to {config.CHROMA_DIRECTORY}/ ({config.COLLECTION_NAME})")
