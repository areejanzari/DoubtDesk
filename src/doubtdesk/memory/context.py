import asyncio

from doubtdesk.memory.store import get_user_recent_exchanges, get_user_eligible_exchanges_by_ids

RECENT_LIMIT = 2
MEMORY_LIMIT = 3


def _compose(recent, hits):
    selected = hits + recent
    newest_id = max((item["id"] for item in selected), default=None)
    return [{"question": item["question"], "answer_excerpt": item["answer"][:1200 if item["id"] == newest_id else 300]} for item in selected]


def _filter_hits(hits, doc_ids):
    return [item for item in hits if not doc_ids or set(doc_ids).intersection(item["doc_ids"])][:MEMORY_LIMIT]


def build_history(conversation_id, user_id, question, doc_ids, *, model=None, search=None):
    recent = get_user_recent_exchanges(conversation_id, user_id, limit=RECENT_LIMIT, eligible_only=True)
    recent_ids = {item["id"] for item in recent}
    hits = []
    if model is not None:
        if search is None:
            from doubtdesk.memory.vector import search_memory
            search = search_memory
        matches = search(conversation_id=conversation_id, user_id=user_id, question=question, doc_ids=doc_ids,
                         model=model, limit=MEMORY_LIMIT, exclude_ids=recent_ids)
        hit_ids = list(dict.fromkeys(item["exchange_id"] for item in matches if item["exchange_id"] not in recent_ids))
        hits = get_user_eligible_exchanges_by_ids(conversation_id, user_id, hit_ids)
        hits = _filter_hits(hits, doc_ids)
    return _compose(recent, hits)


async def abuild_history(conversation_id, user_id, question, doc_ids, *, model=None, search=None):
    recent_task = asyncio.to_thread(get_user_recent_exchanges, conversation_id, user_id, limit=RECENT_LIMIT, eligible_only=True)
    if model is None:
        return _compose(await recent_task, [])
    if search is None:
        from doubtdesk.memory.vector import search_memory
        search = search_memory
    recent, matches = await asyncio.gather(recent_task, asyncio.to_thread(
        search, conversation_id=conversation_id, user_id=user_id, question=question, doc_ids=doc_ids,
        model=model, limit=MEMORY_LIMIT + RECENT_LIMIT, exclude_ids=set(),
    ))
    recent_ids = {item["id"] for item in recent}
    hit_ids = list(dict.fromkeys(item["exchange_id"] for item in matches if item["exchange_id"] not in recent_ids))[:MEMORY_LIMIT]
    hits = await asyncio.to_thread(get_user_eligible_exchanges_by_ids, conversation_id, user_id, hit_ids)
    return _compose(recent, _filter_hits(hits, doc_ids))
