import asyncio


DEFAULT_LIMIT = 5

_limit = DEFAULT_LIMIT
_semaphore = asyncio.Semaphore(DEFAULT_LIMIT)


def set_limit(limit: int):
    """Configure the maximum number of concurrent translation tasks."""
    global _limit, _semaphore
    _limit = max(1, limit)
    _semaphore = asyncio.Semaphore(_limit)


def get_limit() -> int:
    return _limit


class TranslationSlot:
    def __init__(self, semaphore: asyncio.Semaphore):
        self._semaphore = semaphore

    async def __aenter__(self):
        await self._semaphore.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._semaphore.release()


def translation_slot() -> TranslationSlot:
    """Return an async context manager that reserves one concurrency slot."""
    return TranslationSlot(_semaphore)
