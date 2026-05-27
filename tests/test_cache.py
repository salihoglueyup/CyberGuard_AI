"""
CyberGuard AI - Cache Utility Tests
TTL cache, invalidation, sync/async wrapper testleri
"""

import asyncio
import time

import pytest

from app.utils.cache import (
    _CACHE,
    cache_clear_all,
    cache_invalidate,
    cache_stats,
    ttl_cache,
)


@pytest.fixture(autouse=True)
def clear_cache():
    """Her testten önce ve sonra cache'i temizle."""
    cache_clear_all()
    yield
    cache_clear_all()


class TestTTLCacheSync:
    """Senkron TTL cache testleri"""

    def test_sync_cache_miss_and_hit(self):
        """İlk çağrı miss, ikinci çağrı hit döndürmeli"""
        call_count = 0

        @ttl_cache(ttl=60)
        def expensive_fn():
            nonlocal call_count
            call_count += 1
            return {"result": "data", "count": call_count}

        # İlk çağrı — cache miss
        result1 = expensive_fn()
        assert result1["count"] == 1
        assert call_count == 1

        # İkinci çağrı — cache hit
        result2 = expensive_fn()
        assert result2["count"] == 1  # aynı değer
        assert call_count == 1  # fonksiyon tekrar çağrılmadı

    def test_sync_cache_with_args(self):
        """Farklı argümanlar farklı cache entry'leri oluşturmalı"""
        call_count = 0

        @ttl_cache(ttl=60)
        def fn_with_args(x: int, y: int):
            nonlocal call_count
            call_count += 1
            return x + y

        r1 = fn_with_args(1, 2)
        r2 = fn_with_args(3, 4)
        r3 = fn_with_args(1, 2)  # cache hit

        assert r1 == 3
        assert r2 == 7
        assert r3 == 3
        assert call_count == 2  # 3. çağrı cache'den geldi

    def test_sync_cache_expiry(self):
        """TTL süresi dolunca cache yenilenmeli"""
        call_count = 0

        @ttl_cache(ttl=1)  # 1 saniye TTL
        def short_cache_fn():
            nonlocal call_count
            call_count += 1
            return call_count

        result1 = short_cache_fn()
        assert result1 == 1

        time.sleep(1.1)  # TTL dolacak kadar bekle

        result2 = short_cache_fn()
        assert result2 == 2  # Yeni değer — yeniden hesaplandı
        assert call_count == 2

    def test_sync_cache_kwargs(self):
        """Keyword argümanlarla da cache çalışmalı"""
        call_count = 0

        @ttl_cache(ttl=60)
        def fn_with_kwargs(name: str, value: int = 0):
            nonlocal call_count
            call_count += 1
            return f"{name}:{value}"

        r1 = fn_with_kwargs(name="a", value=1)
        r2 = fn_with_kwargs(name="a", value=1)  # hit
        r3 = fn_with_kwargs(name="b", value=2)  # miss

        assert r1 == "a:1"
        assert r2 == "a:1"
        assert r3 == "b:2"
        assert call_count == 2


class TestTTLCacheAsync:
    """Asenkron TTL cache testleri"""

    def test_async_cache_hit(self):
        """Async fonksiyon cache'lenmeli"""
        call_count = 0

        @ttl_cache(ttl=60)
        async def async_fn():
            nonlocal call_count
            call_count += 1
            return {"async": True, "count": call_count}

        async def run():
            r1 = await async_fn()
            r2 = await async_fn()
            return r1, r2

        r1, r2 = asyncio.run(run())
        assert r1["count"] == 1
        assert r2["count"] == 1
        assert call_count == 1

    def test_async_cache_expiry(self):
        """Async cache TTL süresi dolunca yenilenmeli"""
        call_count = 0

        @ttl_cache(ttl=1)
        async def short_async_fn():
            nonlocal call_count
            call_count += 1
            return call_count

        async def run():
            r1 = await short_async_fn()
            await asyncio.sleep(1.1)
            r2 = await short_async_fn()
            return r1, r2

        r1, r2 = asyncio.run(run())
        assert r1 == 1
        assert r2 == 2


class TestCacheInvalidate:
    """Cache geçersizleştirme testleri"""

    def test_invalidate_removes_entries(self):
        """cache_invalidate belirtilen prefix ile entry'leri silmeli"""
        @ttl_cache(ttl=60)
        def my_fn():
            return "value"

        my_fn()  # cache'e ekle
        assert cache_stats()["total_entries"] == 1

        # _make_key uses fn.__qualname__ as prefix
        deleted = cache_invalidate(my_fn.__qualname__)
        assert deleted == 1
        assert cache_stats()["total_entries"] == 0

    def test_invalidate_only_matching_prefix(self):
        """Sadece eşleşen prefix'i sil"""
        @ttl_cache(ttl=60)
        def fn_alpha():
            return "alpha"

        @ttl_cache(ttl=60)
        def fn_beta():
            return "beta"

        fn_alpha()
        fn_beta()
        assert cache_stats()["total_entries"] == 2

        deleted = cache_invalidate(fn_alpha.__qualname__)
        assert deleted == 1
        assert cache_stats()["total_entries"] == 1

    def test_invalidate_nonexistent_prefix(self):
        """Var olmayan prefix 0 döndürmeli"""
        deleted = cache_invalidate("nonexistent_prefix_xyz")
        assert deleted == 0

    def test_cache_clear_all(self):
        """cache_clear_all tüm entry'leri silmeli"""
        @ttl_cache(ttl=60)
        def fn1():
            return 1

        @ttl_cache(ttl=60)
        def fn2():
            return 2

        fn1()
        fn2()
        assert cache_stats()["total_entries"] == 2

        count = cache_clear_all()
        assert count == 2
        assert cache_stats()["total_entries"] == 0


class TestCacheStats:
    """Cache istatistik testleri"""

    def test_stats_empty_cache(self):
        """Boş cache'de tüm sayılar sıfır olmalı"""
        stats = cache_stats()
        assert stats["total_entries"] == 0
        assert stats["valid_entries"] == 0
        assert stats["expired_entries"] == 0

    def test_stats_with_valid_entries(self):
        """Geçerli entry'ler doğru sayılmalı"""
        @ttl_cache(ttl=60)
        def fn():
            return "ok"

        fn()
        stats = cache_stats()
        assert stats["total_entries"] == 1
        assert stats["valid_entries"] == 1
        assert stats["expired_entries"] == 0

    def test_stats_keys(self):
        """cache_stats sözlük döndürmeli"""
        stats = cache_stats()
        assert "total_entries" in stats
        assert "valid_entries" in stats
        assert "expired_entries" in stats
