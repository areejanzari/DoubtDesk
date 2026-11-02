from pathlib import Path
from doubtdesk.memory.store import get_recent_exchanges, get_user_recent_exchanges


def get_conversation_context(
    conversation_id: str,
    question: str,
    doc_ids: list[str],
    limit: int = 3,
) -> list[dict]:
    history = get_recent_exchanges(
        conversation_id,
        limit=30,
    )

    history = [
        exchange
        for exchange in history
        if exchange["status"] != "out_of_scope"
    ]

    if doc_ids:
        selected_ids = set(doc_ids)

        history = [
            exchange
            for exchange in history
            if selected_ids.intersection(exchange["doc_ids"])
        ]

    selected = history[-limit:]

    context = [
        {"question": exchange["question"]}
        for exchange in selected
    ]

    if selected:
        context[-1]["answer_excerpt"] = selected[-1]["answer"][:1200]

    return context


def get_user_conversation_context(
    conversation_id: str,
    user_id: str,
    question: str,
    doc_ids: list[str],
    limit: int = 3,
    db_path: Path = Path("data/conversations.db"),
) -> list[dict]:
    history = get_user_recent_exchanges(
        conversation_id=conversation_id,
        user_id=user_id,
        limit=30,
        db_path=db_path,
    )

    history = [
        exchange
        for exchange in history
        if exchange["status"] != "out_of_scope"
    ]

    if doc_ids:
        selected_ids = set(doc_ids)

        history = [
            exchange
            for exchange in history
            if selected_ids.intersection(exchange["doc_ids"])
        ]

    selected = history[-limit:]

    context = [
        {"question": exchange["question"]}
        for exchange in selected
    ]

    if selected:
        context[-1]["answer_excerpt"] = selected[-1]["answer"][:1200]

    return context
