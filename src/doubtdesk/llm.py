import asyncio
import contextvars
import inspect
import os
import re
import sys
from contextlib import asynccontextmanager

_END = object()
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


async def acall(method, /, *args, **kwargs):
    if inspect.iscoroutinefunction(method):
        return await method(*args, **kwargs)
    result = await asyncio.to_thread(method, *args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def json_reply_instruction(model_cls) -> str:
    import json
    schema = json.dumps(model_cls.model_json_schema(), ensure_ascii=False)
    return "Respond with only a JSON object that matches this schema, with no other text:\n" + schema


def parse_json_reply(text: str | None, model_cls):
    match = _JSON_OBJECT.search(_THINK.sub("", text or ""))
    if match is None:
        raise ValueError("Model returned no JSON object")
    return model_cls.model_validate_json(match.group(0))


def bedrock_credentials(config, purpose: str) -> tuple[str, str]:
    region = config.aws_region or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    if not region:
        raise RuntimeError(f"AWS_REGION is required when {purpose}")
    token = os.getenv("AWS_BEARER_TOKEN_BEDROCK") or os.getenv("ANTHROPIC_AWS_API_KEY")
    if not token:
        raise RuntimeError(f"AWS_BEARER_TOKEN_BEDROCK is required when {purpose}")
    return region, token


def bedrock_chat_client(config, purpose: str):
    region, token = bedrock_credentials(config, purpose)
    from openai import AsyncOpenAI
    return AsyncOpenAI(base_url=f"https://bedrock-mantle.{region}.api.aws/v1", api_key=token, max_retries=2)


SDK_RESOURCES = (("chat", "completions"), ("responses",), ("messages",))


def preload_sdk_resources(*clients) -> None:
    for client in clients:
        if client is None:
            continue
        for path in SDK_RESOURCES:
            target = client
            for name in path:
                if name not in dir(type(target)):
                    break
                target = getattr(target, name)


async def chat_completion(client, **kwargs):
    response = client.chat.completions.create(**kwargs)
    if inspect.isawaitable(response):
        response = await response
    return response


class _ThreadedStream:
    def __init__(self, stream):
        self.stream = stream
        self.iterator = iter(stream)

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await asyncio.to_thread(next, self.iterator, _END)
        if item is _END:
            raise StopAsyncIteration
        return item

    async def get_final_response(self):
        return await asyncio.to_thread(self.stream.get_final_response)


@asynccontextmanager
async def response_stream(client, **kwargs):
    manager = client.responses.stream(**kwargs)
    if hasattr(manager, "__enter__"):
        stream = await asyncio.to_thread(manager.__enter__)
        try:
            yield _ThreadedStream(stream)
        except BaseException:
            if not await asyncio.to_thread(manager.__exit__, *sys.exc_info()):
                raise
        else:
            await asyncio.to_thread(manager.__exit__, None, None, None)
    else:
        async with manager as stream:
            yield stream


class _Loop:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.context = contextvars.copy_context()

    def run(self, awaitable):
        return self.loop.run_until_complete(self.loop.create_task(awaitable, context=self.context))

    def close(self):
        try:
            self.loop.run_until_complete(self.loop.shutdown_default_executor())
        finally:
            self.loop.close()


def run_coroutine(coroutine):
    runner = _Loop()
    try:
        return runner.run(coroutine)
    finally:
        runner.close()


def iterate_with_result(agen, result_event="result"):
    runner = _Loop()
    result = None
    try:
        while True:
            try:
                event = runner.run(agen.__anext__())
            except StopAsyncIteration:
                return result
            if event[0] == result_event:
                result = event[1]
                continue
            yield event
    finally:
        try:
            runner.run(agen.aclose())
        finally:
            runner.close()
