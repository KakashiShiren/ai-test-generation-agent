"""Embedding and ChromaDB retrieval utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.errors import NotFoundError
from chromadb.api.models.Collection import Collection
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


DEFAULT_COLLECTION_NAME = "code_chunks"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHROMA_DIR = Path("data/chroma")


def get_openai_client() -> OpenAI:
    """Create an OpenAI client, relying on OPENAI_API_KEY from the environment."""

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for embedding.")
    return OpenAI()


def get_collection(
    persist_dir: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Collection:
    """Return a local persistent Chroma collection."""

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


@retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(6))
def embed_texts(
    texts: list[str],
    model: str = DEFAULT_EMBEDDING_MODEL,
    client: OpenAI | None = None,
) -> list[list[float]]:
    """Embed a batch of texts with retry and exponential backoff."""

    openai_client = client or get_openai_client()
    response = openai_client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def load_chunks(chunks_path: Path) -> list[dict[str, Any]]:
    """Load chunk dictionaries produced by the AST ingestion phase."""

    return json.loads(chunks_path.read_text(encoding="utf-8"))


def chunk_to_metadata(chunk: dict[str, Any], index: int) -> dict[str, str | int | None]:
    """Convert a chunk dict into Chroma-compatible scalar metadata."""

    return {
        "chunk_index": index,
        "name": chunk["name"],
        "qualified_name": chunk["qualified_name"],
        "kind": chunk["kind"],
        "source_code": chunk["source_code"],
        "docstring": chunk["docstring"],
        "type_hints_json": json.dumps(chunk["type_hints"], sort_keys=True),
        "file_path": chunk["file_path"],
        "start_line": chunk["start_line"],
        "end_line": chunk["end_line"],
    }


def build_vector_store(
    chunks_path: Path = Path("data/chunks/requests_chunks.json"),
    persist_dir: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    batch_size: int = 64,
    reset: bool = False,
) -> int:
    """Embed ingested chunks and store them in a persistent local ChromaDB collection."""

    chunks = load_chunks(chunks_path)
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    if reset:
        try:
            client.delete_collection(collection_name)
        except (ValueError, NotFoundError):
            pass
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    openai_client = get_openai_client()
    added = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        ids = [f"{chunk['file_path']}:{chunk['start_line']}:{chunk['qualified_name']}" for chunk in batch]
        documents = [chunk["embedding_input"] for chunk in batch]
        metadatas = [chunk_to_metadata(chunk, index=start + offset) for offset, chunk in enumerate(batch)]
        embeddings = embed_texts(documents, client=openai_client)
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        added += len(batch)
    return added


def retrieve_context(
    function_signature: str,
    persist_dir: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve top matching chunks for a query function signature."""

    collection = get_collection(persist_dir=persist_dir, collection_name=collection_name)
    query_embedding = embed_texts([function_signature])[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    contexts: list[dict[str, Any]] = []
    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    for item_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        metadata = metadata or {}
        contexts.append(
            {
                "id": item_id,
                "score": 1 - float(distance),
                "embedding_input": document,
                "name": metadata.get("name"),
                "qualified_name": metadata.get("qualified_name"),
                "kind": metadata.get("kind"),
                "source_code": metadata.get("source_code"),
                "docstring": metadata.get("docstring"),
                "type_hints": json.loads(str(metadata.get("type_hints_json") or "{}")),
                "file_path": metadata.get("file_path"),
                "start_line": metadata.get("start_line"),
                "end_line": metadata.get("end_line"),
            }
        )
    return contexts
