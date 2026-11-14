from time import perf_counter
from contextvars import ContextVar
import sys
import logging
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


_active_generation = ContextVar("doubtdesk_active_generation", default=None)
_logger = logging.getLogger(__name__)


@contextmanager
def _safe_context(factory):
    try:
        manager = factory()
        observation = manager.__enter__()
    except Exception:
        _logger.exception("Langfuse observation setup failed")
        yield None
        return
    try:
        yield observation
    except BaseException:
        try:
            manager.__exit__(*sys.exc_info())
        except Exception:
            _logger.exception("Langfuse observation cleanup failed")
        raise
    else:
        try:
            manager.__exit__(None, None, None)
        except Exception:
            _logger.exception("Langfuse observation cleanup failed")


def update_observation(observation, **data):
    if observation is not None:
        try:
            observation.update(**data)
        except Exception:
            _logger.exception("Langfuse observation update failed")


@contextmanager
def trace_stage(name, *, as_type="span", **data):
    if not langfuse_enabled():
        yield None
        return

    def create():
        from langfuse import get_client
        return get_client().start_as_current_observation(name=name, as_type=as_type, **data)

    with _safe_context(create) as observation:
        try:
            yield observation
        except Exception as exc:
            update_observation(observation, level="ERROR", status_message=type(exc).__name__)
            raise


@contextmanager
def trace_generation(name, model, request_id=None, *, input=None):
    started = perf_counter()
    with trace_stage(name, as_type="generation", model=model, input=input,
                     metadata={"request_id": request_id} if request_id else {}) as observation:
        token = _active_generation.set((name, observation, started))
        try:
            yield observation
        finally:
            _active_generation.reset(token)


def _response_output(response):
    raw = getattr(response, "output_text", None)
    if isinstance(raw, str) and raw:
        return raw
    parsed = getattr(response, "output_parsed", None)
    if hasattr(parsed, "model_dump"):
        return parsed.model_dump(mode="json")
    return parsed if isinstance(parsed, (str, dict, list)) else (raw or "")


def observe_llm_response(name, response, model, request_id=None, conversation_id=None, latency_ms=None):
    active = _active_generation.get()
    matching = active is not None and active[0] == name
    if latency_ms is None and matching:
        latency_ms = round((perf_counter() - active[2]) * 1000)
    if request_id:
        from doubtdesk.analytics.tracker import record_openai_usage
        record_openai_usage(response=response, request_id=request_id, conversation_id=conversation_id,
                            operation=name, model=model, latency_ms=latency_ms)
    if not langfuse_enabled():
        return
    usage = getattr(response, "usage", None)
    data = {"output": _response_output(response)}
    if usage is not None:
        data["usage_details"] = {
            "input": getattr(usage, "input_tokens", 0) or 0,
            "output": getattr(usage, "output_tokens", 0) or 0,
        }
    if matching:
        update_observation(active[1], **data)
    else:
        with trace_stage(name, as_type="generation", model=model,
                         metadata={"request_id": request_id} if request_id else {}) as observation:
            update_observation(observation, **data)
