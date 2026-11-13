import json
import logging
from pathlib import Path
import chromadb
from doubtdesk.ingestion.embeddings import load_embedding_model

logger = logging.getLogger(__name__)
MEMORY_COLLECTION = "doubtdesk_conversation_memory"


def get_memory_collection(store_path: Path = Path("data/chroma")):
    client = chromadb.PersistentClient(path=str(store_path))
    return client.get_or_create_collection(
        name=MEMORY_COLLECTION,
        configuration={"hnsw": {"space": "cosine"}},
    )


def add_exchange_memory(
    *, conversation_id: str, user_id: str, exchange_id: int,
    question: str, answer: str, doc_ids: list[str], model=None, collection=None,
):
    if model is None:
        model = load_embedding_model()
    if collection is None:
        collection = get_memory_collection()
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
    doc_ids: list[str], limit: int = 3, model=None, collection=None,
) -> list[dict]:
    if not conversation_id or not user_id or not question.strip():
        return []
    if model is None:
        model = load_embedding_model()
    if collection is None:
        collection = get_memory_collection()
    where = {"$and": [
        {"conversation_id": conversation_id},
        {"user_id": user_id},
    ]}
    if collection.count() == 0:
        return []
    result = collection.query(
        query_embeddings=model.encode([question], normalize_embeddings=True).tolist(),
        n_results=min(20, max(1, collection.count())),
        where=where,
        include=["metadatas", "distances"],
    )
    matched = []
    for metadata in result["metadatas"][0]:
        prior_doc_ids = set(json.loads(metadata["doc_ids"]))
        if doc_ids and not prior_doc_ids.intersection(doc_ids):
            continue
        matched.append({
            "question": metadata["question"],
            "answer_excerpt": metadata["answer_excerpt"],
        })
        if len(matched) == limit:
            break
    return matched


def delete_session_memories(conversation_ids: list[str], collection=None):
    if not conversation_ids:
        return
    if collection is None:
        collection = get_memory_collection()
    for conversation_id in conversation_ids:
        collection.delete(where={"conversation_id": conversation_id})
