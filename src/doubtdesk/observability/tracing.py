import os
from contextlib import contextmanager


def langfuse_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


@contextmanager
def trace_run(name: str, *, user_id: str | None = None, session_id: str | None = None):
    if not langfuse_enabled():
        yield None
        return
    from langfuse import get_client, propagate_attributes
    with get_client().start_as_current_observation(
        as_type="chain", name=name, input={"session_id_present": bool(session_id)}
    ) as observation:
        with propagate_attributes(user_id=user_id, session_id=session_id, tags=["doubtdesk"]):
            yield observation


def graph_config():
    if not langfuse_enabled():
        return None
    from langfuse.langchain import CallbackHandler
    return {"callbacks": [CallbackHandler()]}


def flush_traces():
    if langfuse_enabled():
        from langfuse import get_client
        get_client().flush()


def observe_llm_response(name, response, model, request_id=None, conversation_id=None, latency_ms=None):
    if request_id:
        from doubtdesk.analytics.tracker import record_openai_usage
        record_openai_usage(response=response, request_id=request_id, conversation_id=conversation_id,
                            operation=name, model=model, latency_ms=latency_ms)
    if not langfuse_enabled():
        return
    try:
        from langfuse import get_client
        usage = getattr(response, "usage", None)
        counts = {}
        if usage is not None:
            counts = {
                "input": getattr(usage, "input_tokens", 0) or 0,
                "output": getattr(usage, "output_tokens", 0) or 0,
            }
        with get_client().start_as_current_observation(
            as_type="generation", name=name, model=model,
            metadata={"request_id": request_id} if request_id else {},
        ) as observation:
            if counts:
                observation.update(usage_details=counts)
    except Exception:
        import logging
        logging.getLogger(__name__).exception("Langfuse generation observation failed")
