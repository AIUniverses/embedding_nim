"""Advanced caching system for embedding service."""

import hashlib
import json
import logging
import time
import asyncio
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod
import numpy as np

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import diskcache
    DISKCACHE_AVAILABLE = True
except ImportError:
    DISKCACHE_AVAILABLE = False


class _JSONEncoder(json.JSONEncoder):
    """JSON encoder that understands numpy scalars and arrays."""

    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return super().default(o)


def serialize_value(value: Any) -> bytes:
    """Serialize a cache value to JSON bytes.

    JSON is used instead of pickle so that a compromised or shared cache cannot
    execute arbitrary code during deserialization.
    """
    return json.dumps(value, cls=_JSONEncoder).encode('utf-8')


def deserialize_value(data: bytes) -> Any:
    """Deserialize a cache value from JSON bytes."""
    return json.loads(data.decode('utf-8'))


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete key from cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache backend."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.default_ttl = ttl
        self.logger = logging.getLogger(__name__)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        if time.time() > entry['expires_at']:
            await self.delete(key)
            return None
        
        entry['access_count'] += 1
        entry['last_accessed'] = time.time()
        return entry['value']
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in memory cache."""
        if len(self.cache) >= self.max_size:
            await self._evict_lru()
        
        expires_at = time.time() + (ttl or self.default_ttl)
        self.cache[key] = {
            'value': value,
            'expires_at': expires_at,
            'created_at': time.time(),
            'last_accessed': time.time(),
            'access_count': 0
        }
    
    async def delete(self, key: str) -> None:
        """Delete key from memory cache."""
        self.cache.pop(key, None)
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if key not in self.cache:
            return False
        
        entry = self.cache[key]
        if time.time() > entry['expires_at']:
            await self.delete(key)
            return False
        
        return True
    
    async def _evict_lru(self) -> None:
        """Evict least recently used entries."""
        if not self.cache:
            return
        
        # Remove expired entries first
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time > entry['expires_at']
        ]
        
        for key in expired_keys:
            await self.delete(key)
        
        # If still over capacity, remove LRU entries
        if len(self.cache) >= self.max_size:
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: x[1]['last_accessed']
            )
            
            # Remove oldest 25% of entries
            remove_count = max(1, len(sorted_entries) // 4)
            for key, _ in sorted_entries[:remove_count]:
                await self.delete(key)


class RedisCache(CacheBackend):
    """Redis cache backend."""
    
    def __init__(self, redis_url: str, ttl: int = 3600, compression: bool = False):
        if not REDIS_AVAILABLE:
            raise ImportError("redis package not available")
        
        self.redis_url = redis_url
        self.default_ttl = ttl
        self.compression = compression
        self.redis: Optional[aioredis.Redis] = None
        self.logger = logging.getLogger(__name__)
    
    async def _get_redis(self) -> aioredis.Redis:
        """Get Redis connection."""
        if self.redis is None:
            self.redis = aioredis.from_url(self.redis_url)
        return self.redis
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        try:
            redis = await self._get_redis()
            data = await redis.get(key)
            
            if data is None:
                return None
            
            # Deserialize data
            try:
                value = deserialize_value(data)
            except (ValueError, UnicodeDecodeError):
                # Entry written by an older, unsupported serialization format
                self.logger.warning("Discarding unreadable cache entry: %s", key)
                await self.delete(key)
                return None
            
            # Update access statistics
            await redis.hincrby(f"{key}:stats", "access_count", 1)
            await redis.hset(f"{key}:stats", "last_accessed", time.time())
            
            return value
            
        except Exception as e:
            self.logger.error(f"Redis get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in Redis cache."""
        try:
            redis = await self._get_redis()
            
            # Serialize data
            data = serialize_value(value)
            
            # Set with TTL
            ttl_seconds = ttl or self.default_ttl
            await redis.setex(key, ttl_seconds, data)
            
            # Set statistics
            stats = {
                "created_at": time.time(),
                "last_accessed": time.time(),
                "access_count": 0
            }
            await redis.hset(f"{key}:stats", mapping=stats)
            await redis.expire(f"{key}:stats", ttl_seconds)
            
        except Exception as e:
            self.logger.error(f"Redis set error: {e}")
    
    async def delete(self, key: str) -> None:
        """Delete key from Redis cache."""
        try:
            redis = await self._get_redis()
            await redis.delete(key)
            await redis.delete(f"{key}:stats")
        except Exception as e:
            self.logger.error(f"Redis delete error: {e}")
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        try:
            redis = await self._get_redis()
            await redis.flushdb()
        except Exception as e:
            self.logger.error(f"Redis clear error: {e}")
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        try:
            redis = await self._get_redis()
            return bool(await redis.exists(key))
        except Exception as e:
            self.logger.error(f"Redis exists error: {e}")
            return False


class DiskCache(CacheBackend):
    """Disk cache backend using diskcache."""
    
    def __init__(self, cache_dir: str = "cache", max_size: int = 1024 * 1024 * 1024, ttl: int = 3600):
        if not DISKCACHE_AVAILABLE:
            raise ImportError("diskcache package not available")
        
        self.cache = diskcache.Cache(cache_dir, size_limit=max_size)
        self.default_ttl = ttl
        self.logger = logging.getLogger(__name__)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        try:
            return self.cache.get(key)
        except Exception as e:
            self.logger.error(f"Disk cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in disk cache."""
        try:
            expire_time = time.time() + (ttl or self.default_ttl)
            self.cache.set(key, value, expire=expire_time)
        except Exception as e:
            self.logger.error(f"Disk cache set error: {e}")
    
    async def delete(self, key: str) -> None:
        """Delete key from disk cache."""
        try:
            self.cache.delete(key)
        except Exception as e:
            self.logger.error(f"Disk cache delete error: {e}")
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        try:
            self.cache.clear()
        except Exception as e:
            self.logger.error(f"Disk cache clear error: {e}")
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in disk cache."""
        try:
            return key in self.cache
        except Exception as e:
            self.logger.error(f"Disk cache exists error: {e}")
            return False


class EmbeddingCache:
    """High-level embedding cache manager."""
    
    def __init__(self, backend: CacheBackend):
        self.backend = backend
        self.logger = logging.getLogger(__name__)
        self.hit_count = 0
        self.miss_count = 0
        self.error_count = 0
    
    def _generate_cache_key(self, inputs: List[str], model: str, **kwargs) -> str:
        """Generate cache key for embedding request."""
        # Create a deterministic key from inputs and parameters
        cache_data = {
            'inputs': inputs,
            'model': model,
            **{k: v for k, v in kwargs.items() if v is not None}
        }
        
        # Sort to ensure consistency
        cache_string = json.dumps(cache_data, sort_keys=True)
        
        # Generate hash
        return f"embed:{hashlib.sha256(cache_string.encode()).hexdigest()[:16]}"
    
    async def get_embeddings(
        self,
        inputs: List[str],
        model: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Get cached embeddings."""
        try:
            cache_key = self._generate_cache_key(inputs, model, **kwargs)
            result = await self.backend.get(cache_key)
            
            if result is not None:
                self.hit_count += 1
                self.logger.debug(f"Cache hit for key: {cache_key}")
                return result
            else:
                self.miss_count += 1
                self.logger.debug(f"Cache miss for key: {cache_key}")
                return None
                
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Cache get error: {e}")
            return None
    
    async def set_embeddings(
        self,
        inputs: List[str],
        model: str,
        embeddings: Dict[str, Any],
        ttl: Optional[int] = None,
        **kwargs
    ) -> None:
        """Cache embeddings."""
        try:
            cache_key = self._generate_cache_key(inputs, model, **kwargs)
            await self.backend.set(cache_key, embeddings, ttl)
            self.logger.debug(f"Cached embeddings for key: {cache_key}")
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Cache set error: {e}")
    
    async def invalidate_model(self, model: str) -> None:
        """Invalidate all cache entries for a specific model."""
        # This is a simple implementation - in production you might want
        # to use key patterns or tags for more efficient invalidation
        self.logger.warning(f"Model cache invalidation not implemented for {model}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests) if total_requests > 0 else 0
        
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'error_count': self.error_count,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }
    
    async def clear_cache(self) -> None:
        """Clear all cache entries."""
        await self.backend.clear()
        self.hit_count = 0
        self.miss_count = 0
        self.error_count = 0


def create_cache_backend(backend_type: str, **kwargs) -> CacheBackend:
    """Factory function to create cache backend."""
    if backend_type == "memory":
        return MemoryCache(**kwargs)
    elif backend_type == "redis":
        return RedisCache(**kwargs)
    elif backend_type == "disk":
        return DiskCache(**kwargs)
    else:
        raise ValueError(f"Unknown cache backend type: {backend_type}")


def create_embedding_cache(config) -> EmbeddingCache:
    """Create embedding cache from configuration."""
    backend_type = config.get_cache_backend()
    
    if backend_type == "memory":
        backend = MemoryCache(
            max_size=config.caching.max_size,
            ttl=config.caching.ttl_seconds
        )
    elif backend_type == "redis":
        backend = RedisCache(
            redis_url=config.get_redis_url(),
            ttl=config.caching.ttl_seconds,
            compression=config.caching.compression
        )
    elif backend_type == "disk":
        backend = DiskCache(
            ttl=config.caching.ttl_seconds
        )
    else:
        raise ValueError(f"Unknown cache backend: {backend_type}")
    
    return EmbeddingCache(backend)
