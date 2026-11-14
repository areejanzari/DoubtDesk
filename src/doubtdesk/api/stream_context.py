from contextvars import copy_context

import anyio
from starlette.responses import StreamingResponse


_END = object()


class _ContextIterator:
    def __init__(self, content):
        self.iterator = iter(content)
        self.context = copy_context()
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.closed:
            raise StopAsyncIteration
        item = await anyio.to_thread.run_sync(self.context.run, next, self.iterator, _END)
        if item is _END:
            await self.aclose()
            raise StopAsyncIteration
        return item

    async def aclose(self):
        if self.closed:
            return
        self.closed = True
        close = getattr(self.iterator, "close", None)
        if close is not None:
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(self.context.run, close)


class ContextStreamingResponse(StreamingResponse):
    def __init__(self, content, *, on_close=None, **kwargs):
        self.on_close = on_close
        self.context_iterator = _ContextIterator(content)
        super().__init__(self.context_iterator, **kwargs)

    async def stream_response(self, send):
        try:
            await super().stream_response(send)
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    await self.context_iterator.aclose()
                finally:
                    if self.on_close is not None:
                        await anyio.to_thread.run_sync(self.on_close)
