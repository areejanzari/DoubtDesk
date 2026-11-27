import asyncio
import json
from time import perf_counter
from pydantic import BaseModel, Field, field_validator
from doubtdesk.llm import acall, chat_completion, json_reply_instruction, parse_json_reply, run_coroutine
from doubtdesk.observability.tracing import observe_llm_response, trace_generation, update_observation


class ParaphraseResult(BaseModel):
    questions: list[str] = Field(min_length=3, max_length=3)

    @field_validator("questions")
    @classmethod
    def validate_questions(cls, questions: list[str]) -> list[str]:
        if any(not question.strip() for question in questions):
            raise ValueError("Paraphrases cannot be empty.")
        return questions


class ContextualParaphraseResult(ParaphraseResult):
    resolved_question: str = Field(min_length=1)


INSTRUCTION = (
    "You resolve conversational student questions for DoubtDesk. "
    "Use prior conversation context to resolve explicit references and elliptical follow-ups. "
    "When the current question leaves its subject implicit (for example 'any other?', 'why?', "
    "'give an example'), carry over the specific subject of the most recent exchange, keeping its "
    "qualifiers (such as the particular organism, process, or substance) rather than generalising it. "
    "If the current question is already self-contained or changes topic, do not add earlier context. "
    "Rewrite the question so it stands alone, preserving intent. "
    "Return exactly three distinct paraphrases. "
    "Correct spelling without changing meaning; the third paraphrase focuses on the core concept. "
    "Do not turn an off-topic question into an academic question. "
    "Do not invent missing information or answer the question."
)


def _paraphrase_request(model_name: str, question: str, history: list[dict]) -> dict:
    if not question.strip():
        raise ValueError("Question cannot be empty")
    return {
        "model": model_name,
        "input": [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": json.dumps({"history": history, "current_question": question})},
        ],
        "text_format": ContextualParaphraseResult,
        "reasoning": {"effort": "none"},
        "max_output_tokens": 650,
    }


def _paraphrase_result(response) -> ContextualParaphraseResult:
    if response.output_parsed is None:
        raise ValueError("Model returned no valid paraphrases")
    return ContextualParaphraseResult.model_validate(response.output_parsed)


DEFAULT_BEDROCK_PARAPHRASE_MODEL = "qwen.qwen3-235b-a22b-2507"


class BedrockChat:
    provider = "bedrock_openai"

    def __init__(self, client):
        self.client = client

    async def close(self):
        await self.client.close()


def build_paraphraser(config, llm_client, model_name: str):
    if config.paraphrase_provider == "bedrock_openai":
        from doubtdesk.llm import bedrock_chat_client
        chat = BedrockChat(bedrock_chat_client(config, "PARAPHRASE_PROVIDER is bedrock_openai"))
        return chat, config.paraphrase_model or DEFAULT_BEDROCK_PARAPHRASE_MODEL
    return llm_client, config.paraphrase_model or model_name


def _chat_request(model_name: str, question: str, history: list[dict]) -> dict:
    if not question.strip():
        raise ValueError("Question cannot be empty")
    return {
        "model": model_name,
        "messages": [
            {"role": "system", "content": INSTRUCTION + "\n\n" + json_reply_instruction(ContextualParaphraseResult)},
            {"role": "user", "content": json.dumps({"history": history, "current_question": question})},
        ],
        "max_tokens": 650,
    }


def _record_chat_usage(chat, response, model_name, request_id, conversation_id, started, error=None):
    if not request_id:
        return
    from doubtdesk.analytics.tracker import record_anthropic_failure, record_anthropic_usage
    common = dict(response=response, request_id=request_id, conversation_id=conversation_id, operation="paraphrasing",
                  model=model_name, latency_ms=round((perf_counter() - started) * 1000), provider=chat.provider)
    if error is None:
        record_anthropic_usage(**common)
    else:
        record_anthropic_failure(**common, error_type=type(error).__name__)


async def _achat_paraphrase(chat, model_name, question, history, request_id, conversation_id):
    request = _chat_request(model_name, question, history)
    with trace_generation("paraphrasing", model_name, request_id,
                          input={"question": question, "history_count": len(history)}) as observation:
        started = perf_counter()
        response = None
        try:
            response = await chat_completion(chat.client, **request)
            choice = response.choices[0]
            if getattr(choice, "finish_reason", None) == "length":
                raise ValueError("Paraphrasing stopped early: length")
            result = parse_json_reply(choice.message.content, ContextualParaphraseResult)
        except Exception as exc:
            await asyncio.to_thread(_record_chat_usage, chat, response, model_name, request_id, conversation_id, started, exc)
            raise
        await asyncio.to_thread(_record_chat_usage, chat, response, model_name, request_id, conversation_id, started)
        usage = getattr(response, "usage", None)
        update_observation(observation, output=result.model_dump(mode="json"), usage_details={
            "input": getattr(usage, "prompt_tokens", 0) or 0,
            "output": getattr(usage, "completion_tokens", 0) or 0,
        })
    return result


def resolve_and_paraphrase(
    client,
    model_name: str,
    question: str,
    history: list[dict],
    *,
    request_id: str | None = None,
    conversation_id: str | None = None,
) -> ContextualParaphraseResult:
    if isinstance(client, BedrockChat):
        return run_coroutine(_achat_paraphrase(client, model_name, question, history, request_id, conversation_id))
    request = _paraphrase_request(model_name, question, history)
    with trace_generation("paraphrasing", model_name, request_id, input={"question": question, "history_count": len(history)}):
        response = client.responses.parse(**request)
        observe_llm_response("paraphrasing", response, model_name, request_id, conversation_id)
    return _paraphrase_result(response)


async def aresolve_and_paraphrase(
    client,
    model_name: str,
    question: str,
    history: list[dict],
    *,
    request_id: str | None = None,
    conversation_id: str | None = None,
) -> ContextualParaphraseResult:
    if isinstance(client, BedrockChat):
        return await _achat_paraphrase(client, model_name, question, history, request_id, conversation_id)
    request = _paraphrase_request(model_name, question, history)
    with trace_generation("paraphrasing", model_name, request_id, input={"question": question, "history_count": len(history)}):
        response = await acall(client.responses.parse, **request)
        await asyncio.to_thread(observe_llm_response, "paraphrasing", response, model_name, request_id, conversation_id)
    return _paraphrase_result(response)
