"""
CyberGuard AI - Cache Utility
Hafif in-process TTL cache — Redis gerektirmez.

Kullanım:
    from app.utils.cache import ttl_cache, cache_invalidate

    @ttl_cache(ttl=30)
    def get_dashboard_stats():
        ...  # pahalı işlem

    cache_invalidate("get_dashboard_stats")   # elle temizle
"""

import functools
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# Global cache store: key → (value, expires_at)
_CACHE: dict[str, tuple[Any, float]] = {}
_CACHE_LOCK = threading.Lock()


def _make_key(fn: Callable, args: tuple, kwargs: dict) -> str:
    """Fonksiyon adı + argümanlardan cache anahtarı oluştur."""
    key_parts = [fn.__qualname__]
    if args:
        key_parts.append(str(args))
    if kwargs:
        key_parts.append(str(sorted(kwargs.items())))
    return ":".join(key_parts)


def ttl_cache(ttl: int = 60):
    """
    TTL tabanlı in-process cache dekoratörü.

    Args:
        ttl: Saniye cinsinden cache süresi (varsayılan: 60s)

    Örnek:
        @ttl_cache(ttl=30)
        async def get_stats():
            return expensive_db_call()
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            key = _make_key(fn, args, kwargs)
            with _CACHE_LOCK:
                if key in _CACHE:
                    value, expires_at = _CACHE[key]
                    if time.monotonic() < expires_at:
                        logger.debug(f"[Cache HIT] {key}")
                        return value
                    else:
                        del _CACHE[key]

            result = await fn(*args, **kwargs)

            with _CACHE_LOCK:
                _CACHE[key] = (result, time.monotonic() + ttl)
                logger.debug(f"[Cache SET] {key} (ttl={ttl}s)")

            return result

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            key = _make_key(fn, args, kwargs)
            with _CACHE_LOCK:
                if key in _CACHE:
                    value, expires_at = _CACHE[key]
                    if time.monotonic() < expires_at:
                        logger.debug(f"[Cache HIT] {key}")
                        return value
                    else:
                        del _CACHE[key]

            result = fn(*args, **kwargs)

            with _CACHE_LOCK:
                _CACHE[key] = (result, time.monotonic() + ttl)
                logger.debug(f"[Cache SET] {key} (ttl={ttl}s)")

            return result

        import asyncio
        if asyncio.iscoroutinefunction(fn):
            return async_wrapper
        return sync_wrapper

    return decorator


def cache_invalidate(fn_name_prefix: str) -> int:
    """
    Belirtilen fonksiyon adıyla başlayan tüm cache entry'lerini sil.

    Returns:
        Silinen entry sayısı
    """
    with _CACHE_LOCK:
        keys_to_delete = [k for k in _CACHE if k.startswith(fn_name_prefix)]
        for k in keys_to_delete:
            del _CACHE[k]
    if keys_to_delete:
        logger.debug(f"[Cache INVALIDATE] {len(keys_to_delete)} entry silindi: {fn_name_prefix}*")
    return len(keys_to_delete)


def cache_clear_all() -> int:
    """Tüm cache'i temizle. Genellikle test sonrası kullanılır."""
    with _CACHE_LOCK:
        count = len(_CACHE)
        _CACHE.clear()
    logger.debug(f"[Cache CLEAR ALL] {count} entry silindi")
    return count


def cache_stats() -> dict:
    """Cache durumu hakkında bilgi döndür."""
    now = time.monotonic()
    with _CACHE_LOCK:
        total = len(_CACHE)
        valid = sum(1 for _, (_, exp) in _CACHE.items() if exp > now)
        expired = total - valid
    return {
        "total_entries": total,
        "valid_entries": valid,
        "expired_entries": expired,
    }
