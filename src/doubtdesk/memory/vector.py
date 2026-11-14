from doubtdesk.config import settings, runtime_settings
import json
import math
import logging
from pathlib import Path
import chromadb

logger = logging.getLogger(__name__)
MEMORY_COLLECTION = "doubtdesk_conversation_memory"


def get_memory_collection(store_path: Path = settings.chroma_path, *, client=None):
    if client is None:
        client = chromadb.PersistentClient(path=str(store_path))
    return client.get_or_create_collection(
        name=MEMORY_COLLECTION,
        configuration={"hnsw": {"space": "cosine"}},
    )


def add_exchange_memory(
    *, conversation_id: str, user_id: str, exchange_id: int,
    question: str, answer: str, doc_ids: list[str], model, collection,
):
    text = f"Question: {question}\nAnswer: {answer[:1600]}"
    embedding = model.encode([text], normalize_embeddings=True)[0]
    collection.upsert(
        ids=[f"exchange_{exchange_id}"],
        documents=[text],
        embeddings=[embedding.tolist()],
        metadatas=[{
            "conversation_id": conversation_id,
            "user_id": user_id,
            "doc_ids": json.dumps(doc_ids),
            "question": question[:750],
            "answer_excerpt": answer[:1200],
        }],
    )


def search_memory(
    *, conversation_id: str, user_id: str, question: str,
    doc_ids: list[str], limit: int = 3, model, collection,
    exclude_ids=None, max_distance=None,
) -> list[dict]:
    if not conversation_id or not user_id or not question.strip():
        return []
    where = {"$and": [
        {"conversation_id": conversation_id},
        {"user_id": user_id},
    ]}
    count = collection.count()
    if count == 0 or limit <= 0:
        return []
    max_distance = runtime_settings().memory_max_distance if max_distance is None else max_distance
    if not 0 <= max_distance <= 2:
        raise ValueError("MEMORY_MAX_DISTANCE must be between 0 and 2")
    excluded = set(exclude_ids or ())
    result = collection.query(
        query_embeddings=model.encode([question], normalize_embeddings=True).tolist(),
        n_results=min(20, count),
        where=where,
        include=["metadatas", "distances"],
    )
    matched = []
    for raw_id, metadata, distance in zip(result["ids"][0], result["metadatas"][0], result["distances"][0], strict=True):
        if not raw_id.startswith("exchange_") or not raw_id[9:].isdigit():
            continue
        exchange_id = int(raw_id[9:])
        if exchange_id in excluded or not math.isfinite(distance) or distance > max_distance:
            continue
        if not metadata or metadata.get("conversation_id") != conversation_id or metadata.get("user_id") != user_id:
            continue
        excluded.add(exchange_id)
        prior_doc_ids = set(json.loads(metadata["doc_ids"]))
        if doc_ids and not prior_doc_ids.intersection(doc_ids):
            continue
        matched.append({
            "exchange_id": exchange_id,
            "question": metadata["question"],
            "answer_excerpt": metadata["answer_excerpt"],
        })
        if len(matched) == limit:
            break
    return matched


def delete_session_memories(conversation_ids: list[str], *, collection):
    if not conversation_ids:
        return
    for conversation_id in conversation_ids:
        collection.delete(where={"conversation_id": conversation_id})
