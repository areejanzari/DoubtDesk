import asyncio
import threading
from time import monotonic, sleep


class RequestPacer:
    def __init__(self, rpm: int):
        if rpm < 1:
            raise ValueError("rpm must be at least 1")
        self.interval = 60.0 / rpm
        self._next = 0.0
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            now = monotonic()
            slot = max(now, self._next)
            self._next = slot + self.interval
        if slot > now:
            sleep(slot - now)

    async def acquire_async(self):
        await asyncio.to_thread(self.acquire)
