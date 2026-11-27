import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock

from doubtdesk.config import settings
from doubtdesk.ingestion.registry import STALE_PROCESSING_SECONDS

logger = logging.getLogger(__name__)

_executor: ThreadPoolExecutor | None = None
_lock = Lock()


def submit(function, *args) -> Future:
    global _executor
    with _lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=settings.ingestion_workers, thread_name_prefix="ingestion")
        return _executor.submit(function, *args)


def shutdown() -> None:
    global _executor
    with _lock:
        executor, _executor = _executor, None
    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=True)


def remove_stale_uploads() -> int:
    directory = settings.upload_dir
    if not directory.is_dir():
        return 0
    cutoff = time.time() - STALE_PROCESSING_SECONDS
    removed = 0
    for path in directory.glob("*.pdf"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            logger.warning("Could not remove stale upload %s", path.name)
    return removed
