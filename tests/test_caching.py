"""Unit tests for src/utils/caching.py."""

from types import SimpleNamespace

import pytest

from src.utils import caching
from src.utils.caching import (
    CacheBackend,
    DiskCache,
    EmbeddingCache,
    MemoryCache,
    RedisCache,
    create_cache_backend,
    create_embedding_cache,
)


class FailingBackend(CacheBackend):
    """Backend whose operations always raise, to exercise error handling."""

    async def get(self, key):
        raise RuntimeError("get failed")

    async def set(self, key, value, ttl=None):
        raise RuntimeError("set failed")

    async def delete(self, key):
        raise RuntimeError("delete failed")

    async def clear(self):
        raise RuntimeError("clear failed")

    async def exists(self, key):
        raise RuntimeError("exists failed")


class TestMemoryCache:
    async def test_set_get_and_exists(self):
        cache = MemoryCache(max_size=10, ttl=60)
        await cache.set("a", [1, 2, 3])
        assert await cache.get("a") == [1, 2, 3]
        assert await cache.exists("a") is True
        assert cache.cache["a"]["access_count"] == 1

    async def test_missing_key_returns_none(self):
        cache = MemoryCache()
        assert await cache.get("missing") is None
        assert await cache.exists("missing") is False

    async def test_expired_entry_is_dropped_on_get(self, monkeypatch):
        cache = MemoryCache(ttl=1)
        await cache.set("a", "value")
        monkeypatch.setattr(caching.time, "time", lambda: 10**12)
        assert await cache.get("a") is None
        assert "a" not in cache.cache

    async def test_expired_entry_is_dropped_on_exists(self, monkeypatch):
        cache = MemoryCache(ttl=1)
        await cache.set("a", "value")
        monkeypatch.setattr(caching.time, "time", lambda: 10**12)
        assert await cache.exists("a") is False
        assert "a" not in cache.cache

    async def test_per_call_ttl_overrides_default(self):
        cache = MemoryCache(ttl=1)
        await cache.set("a", "value", ttl=1000)
        assert cache.cache["a"]["expires_at"] - cache.cache["a"][
            "created_at"
        ] == pytest.approx(1000, abs=1)

    async def test_delete_and_clear(self):
        cache = MemoryCache()
        await cache.set("a", 1)
        await cache.set("b", 2)
        await cache.delete("a")
        assert "a" not in cache.cache
        await cache.clear()
        assert cache.cache == {}

    async def test_eviction_removes_expired_entries_first(self, monkeypatch):
        cache = MemoryCache(max_size=2, ttl=10)
        current = [1000.0]
        monkeypatch.setattr(caching.time, "time", lambda: current[0])
        await cache.set("stale", 1)
        current[0] += 100
        await cache.set("fresh", 2)
        await cache.set("new", 3)
        assert "stale" not in cache.cache
        assert set(cache.cache) == {"fresh", "new"}

    async def test_eviction_falls_back_to_lru(self):
        cache = MemoryCache(max_size=4, ttl=3600)
        for index in range(4):
            await cache.set(f"key{index}", index)
        await cache.get("key0")  # refresh access time of the oldest entry
        await cache.set("key4", 4)
        assert "key1" not in cache.cache
        assert "key0" in cache.cache

    async def test_evict_lru_on_empty_cache_is_noop(self):
        cache = MemoryCache()
        await cache._evict_lru()
        assert cache.cache == {}


class TestBackendFactory:
    def test_creates_memory_backend(self):
        backend = create_cache_backend("memory", max_size=5, ttl=30)
        assert isinstance(backend, MemoryCache)
        assert backend.max_size == 5
        assert backend.default_ttl == 30

    def test_creates_disk_backend(self, tmp_path):
        backend = create_cache_backend("disk", cache_dir=str(tmp_path / "c"), ttl=30)
        assert isinstance(backend, DiskCache)

    def test_rejects_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown cache backend type"):
            create_cache_backend("s3")

    def test_redis_backend_requires_redis_package(self, monkeypatch):
        monkeypatch.setattr(caching, "REDIS_AVAILABLE", False)
        with pytest.raises(ImportError, match="redis package not available"):
            create_cache_backend("redis", redis_url="redis://localhost:6379")

    def test_disk_backend_requires_diskcache_package(self, monkeypatch):
        monkeypatch.setattr(caching, "DISKCACHE_AVAILABLE", False)
        with pytest.raises(ImportError, match="diskcache package not available"):
            create_cache_backend("disk")


class TestCreateEmbeddingCache:
    @staticmethod
    def make_config(backend_type: str, redis_url: str = "redis://localhost:6379"):
        return SimpleNamespace(
            get_cache_backend=lambda: backend_type,
            get_redis_url=lambda: redis_url,
            caching=SimpleNamespace(max_size=7, ttl_seconds=11, compression=True),
        )

    def test_memory_backend_uses_config_values(self):
        cache = create_embedding_cache(self.make_config("memory"))
        assert isinstance(cache, EmbeddingCache)
        assert cache.backend.max_size == 7
        assert cache.backend.default_ttl == 11

    def test_redis_backend_uses_config_values(self):
        cache = create_embedding_cache(self.make_config("redis"))
        assert isinstance(cache.backend, RedisCache)
        assert cache.backend.redis_url == "redis://localhost:6379"
        assert cache.backend.default_ttl == 11
        assert cache.backend.compression is True

    def test_disk_backend_uses_config_ttl(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        cache = create_embedding_cache(self.make_config("disk"))
        assert isinstance(cache.backend, DiskCache)
        assert cache.backend.default_ttl == 11

    def test_rejects_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown cache backend"):
            create_embedding_cache(self.make_config("s3"))


class TestDiskCache:
    @pytest.fixture
    def cache(self, tmp_path) -> DiskCache:
        return DiskCache(cache_dir=str(tmp_path / "diskcache"), ttl=60)

    async def test_set_get_exists_delete_clear(self, cache):
        await cache.set("a", {"x": 1})
        assert await cache.get("a") == {"x": 1}
        assert await cache.exists("a") is True
        await cache.delete("a")
        assert await cache.exists("a") is False
        await cache.set("b", 2)
        await cache.clear()
        assert await cache.get("b") is None

    async def test_errors_are_swallowed(self, cache, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("disk failure")

        monkeypatch.setattr(cache.cache, "get", boom)
        monkeypatch.setattr(cache.cache, "set", boom)
        monkeypatch.setattr(cache.cache, "delete", boom)
        monkeypatch.setattr(cache.cache, "clear", boom)
        monkeypatch.setattr(cache.cache, "__contains__", boom)

        assert await cache.get("a") is None
        await cache.set("a", 1)
        await cache.delete("a")
        await cache.clear()
        assert await cache.exists("a") is False


class TestEmbeddingCache:
    @pytest.fixture
    def cache(self) -> EmbeddingCache:
        return EmbeddingCache(MemoryCache(max_size=10, ttl=60))

    def test_cache_key_is_deterministic_and_order_independent(self, cache):
        key = cache._generate_cache_key(["a"], "model", dimensions=8, encoding=None)
        assert key.startswith("embed:")
        assert len(key) == len("embed:") + 16
        assert key == cache._generate_cache_key(
            ["a"], "model", encoding=None, dimensions=8
        )

    def test_cache_key_ignores_none_values(self, cache):
        assert cache._generate_cache_key(["a"], "m", dimensions=None) == (
            cache._generate_cache_key(["a"], "m")
        )

    def test_cache_key_varies_with_inputs_and_model(self, cache):
        base = cache._generate_cache_key(["a"], "m")
        assert base != cache._generate_cache_key(["b"], "m")
        assert base != cache._generate_cache_key(["a"], "other")
        assert base != cache._generate_cache_key(["a"], "m", dimensions=8)

    async def test_hit_and_miss_counters(self, cache):
        assert await cache.get_embeddings(["a"], "m") is None
        await cache.set_embeddings(["a"], "m", {"data": [1]})
        assert await cache.get_embeddings(["a"], "m") == {"data": [1]}

        stats = cache.get_stats()
        assert stats["hit_count"] == 1
        assert stats["miss_count"] == 1
        assert stats["error_count"] == 0
        assert stats["total_requests"] == 2
        assert stats["hit_rate"] == pytest.approx(0.5)

    async def test_kwargs_participate_in_lookup(self, cache):
        await cache.set_embeddings(["a"], "m", {"data": [1]}, dimensions=8)
        assert await cache.get_embeddings(["a"], "m", dimensions=8) == {"data": [1]}
        assert await cache.get_embeddings(["a"], "m", dimensions=16) is None

    async def test_ttl_is_forwarded_to_backend(self, cache):
        await cache.set_embeddings(["a"], "m", {"data": [1]}, ttl=1)
        entry = next(iter(cache.backend.cache.values()))
        assert entry["expires_at"] - entry["created_at"] == pytest.approx(1, abs=1)

    def test_empty_stats_report_zero_hit_rate(self, cache):
        assert cache.get_stats() == {
            "hit_count": 0,
            "miss_count": 0,
            "error_count": 0,
            "hit_rate": 0,
            "total_requests": 0,
        }

    async def test_backend_errors_increment_error_count(self):
        cache = EmbeddingCache(FailingBackend())
        assert await cache.get_embeddings(["a"], "m") is None
        await cache.set_embeddings(["a"], "m", {"data": [1]})
        assert cache.get_stats()["error_count"] == 2

    async def test_clear_cache_resets_counters(self, cache):
        await cache.set_embeddings(["a"], "m", {"data": [1]})
        await cache.get_embeddings(["a"], "m")
        await cache.clear_cache()
        assert cache.backend.cache == {}
        assert cache.get_stats()["total_requests"] == 0

    async def test_invalidate_model_is_a_documented_noop(self, cache, caplog):
        await cache.set_embeddings(["a"], "m", {"data": [1]})
        await cache.invalidate_model("m")
        assert await cache.get_embeddings(["a"], "m") == {"data": [1]}


class TestRedisCache:
    class FakeRedis:
        def __init__(self):
            self.store = {}
            self.hashes = {}
            self.expirations = {}

        async def get(self, key):
            return self.store.get(key)

        async def setex(self, key, ttl, data):
            self.store[key] = data
            self.expirations[key] = ttl

        async def delete(self, key):
            self.store.pop(key, None)
            self.hashes.pop(key, None)

        async def flushdb(self):
            self.store.clear()
            self.hashes.clear()

        async def exists(self, key):
            return 1 if key in self.store else 0

        async def hincrby(self, key, field, amount):
            bucket = self.hashes.setdefault(key, {})
            bucket[field] = bucket.get(field, 0) + amount

        async def hset(self, key, field=None, value=None, mapping=None):
            bucket = self.hashes.setdefault(key, {})
            if mapping:
                bucket.update(mapping)
            else:
                bucket[field] = value

        async def expire(self, key, ttl):
            self.expirations[key] = ttl

    @pytest.fixture
    def cache(self, monkeypatch):
        cache = RedisCache("redis://localhost:6379", ttl=42)
        fake = self.FakeRedis()

        async def get_redis():
            return fake

        monkeypatch.setattr(cache, "_get_redis", get_redis)
        cache.fake = fake
        return cache

    async def test_set_get_exists_delete_clear(self, cache):
        await cache.set("a", {"x": 1})
        assert cache.fake.expirations["a"] == 42
        assert await cache.get("a") == {"x": 1}
        assert cache.fake.hashes["a:stats"]["access_count"] == 1
        assert await cache.exists("a") is True

        await cache.delete("a")
        assert await cache.exists("a") is False

        await cache.set("b", 2, ttl=7)
        assert cache.fake.expirations["b"] == 7
        await cache.clear()
        assert await cache.get("b") is None

    async def test_lazy_connection_is_reused(self, monkeypatch):
        cache = RedisCache("redis://localhost:6379")
        created = []

        def from_url(url):
            created.append(url)
            return self.FakeRedis()

        monkeypatch.setattr(caching.aioredis, "from_url", from_url)
        first = await cache._get_redis()
        second = await cache._get_redis()
        assert first is second
        assert created == ["redis://localhost:6379"]

    async def test_errors_are_swallowed(self, monkeypatch):
        cache = RedisCache("redis://localhost:6379")

        async def boom():
            raise RuntimeError("connection refused")

        monkeypatch.setattr(cache, "_get_redis", boom)
        assert await cache.get("a") is None
        await cache.set("a", 1)
        await cache.delete("a")
        await cache.clear()
        assert await cache.exists("a") is False
