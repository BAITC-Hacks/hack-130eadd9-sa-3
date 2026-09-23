"""Bounded process-local cache of immutable, validated model reasons."""

from collections import OrderedDict
from threading import Lock
from time import monotonic

MAX_ENTRIES = 128
_entries: OrderedDict[str, tuple[float, tuple[str, ...]]] = OrderedDict()
_lock = Lock()
_generation = 0


def clear_cache() -> None:
    global _generation
    with _lock:
        _entries.clear()
        _generation += 1


def generation() -> int:
    with _lock:
        return _generation


def get(key: str) -> tuple[str, ...] | None:
    with _lock:
        item = _entries.get(key)
        if item is None:
            return None
        expires, reasons = item
        if expires <= monotonic():
            del _entries[key]
            return None
        _entries.move_to_end(key)
        return reasons


def put(key: str, reasons: tuple[str, ...], ttl: float, epoch: int) -> None:
    if ttl <= 0:
        return
    with _lock:
        # An in-flight request must not undo an explicit cache clear.
        if epoch != _generation:
            return
        _entries[key] = (monotonic() + ttl, reasons)
        _entries.move_to_end(key)
        while len(_entries) > MAX_ENTRIES:
            _entries.popitem(last=False)
