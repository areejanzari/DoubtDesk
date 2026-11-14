from doubtdesk.config import settings
from doubtdesk.memory.store import get_user_recent_exchanges, get_user_eligible_exchanges_by_ids


def build_history(conversation_id, user_id, question, doc_ids, *, model=None, collection=None,
                  db_path=settings.conversations_db, search=None):
    recent = get_user_recent_exchanges(conversation_id, user_id, limit=2, db_path=db_path, eligible_only=True)
    recent_ids = {item["id"] for item in recent}
    hits = []
    if model is not None and collection is not None:
        if search is None:
            from doubtdesk.memory.vector import search_memory
            search = search_memory
        matches = search(conversation_id=conversation_id, user_id=user_id, question=question, doc_ids=doc_ids,
                         model=model, collection=collection, limit=3, exclude_ids=recent_ids)
        hit_ids = list(dict.fromkeys(item["exchange_id"] for item in matches if item["exchange_id"] not in recent_ids))
        hits = get_user_eligible_exchanges_by_ids(conversation_id, user_id, hit_ids, db_path)
        hits = [item for item in hits if not doc_ids or set(doc_ids).intersection(item["doc_ids"])][:3]
    selected = hits + recent
    newest_id = max((item["id"] for item in selected), default=None)
    return [{"question": item["question"], "answer_excerpt": item["answer"][:1200 if item["id"] == newest_id else 300]} for item in selected]
