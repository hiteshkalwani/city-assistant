"""
Redis client configuration and connection management.

This module provides Redis connection setup and caching utilities.
"""

import json
import logging
from typing import Any, Optional

import redis.asyncio as redis
from redis.exceptions import ConnectionError, RedisError

from app.core.config import settings

logger = logging.getLogger(__name__)


class RedisClient:
    """Redis client wrapper for caching operations."""

    def __init__(self):
        self._redis_pool: Optional[redis.ConnectionPool] = None
        self._sync_redis: Optional[redis.Redis] = None

    async def get_redis_pool(self) -> redis.ConnectionPool:
        """Get or create Redis connection pool."""
        if self._redis_pool is None:
            try:
                self._redis_pool = redis.ConnectionPool.from_url(
                    settings.REDIS_URL,
                    password=settings.REDIS_PASSWORD,
                    db=settings.REDIS_DB,
                    max_connections=settings.REDIS_MAX_CONNECTIONS,
                    socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
                    socket_connect_timeout=settings.REDIS_SOCKET_CONNECT_TIMEOUT,
                    decode_responses=True,
                    encoding="utf-8"
                )
                logger.info("Redis connection pool created successfully")
            except Exception as e:
                logger.error(f"Failed to create Redis connection pool: {e}")
                raise
        return self._redis_pool

    async def get_redis(self) -> redis.Redis:
        """Get Redis connection from pool."""
        pool = await self.get_redis_pool()
        return redis.Redis(connection_pool=pool)

    def get_sync_redis(self) -> redis.Redis:
        """Get synchronous Redis connection."""
        if self._sync_redis is None:
            try:
                import redis as sync_redis
                self._sync_redis = sync_redis.from_url(
                    settings.REDIS_URL,
                    password=settings.REDIS_PASSWORD,
                    db=settings.REDIS_DB,
                    max_connections=settings.REDIS_MAX_CONNECTIONS,
                    socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
                    socket_connect_timeout=settings.REDIS_SOCKET_CONNECT_TIMEOUT,
                    decode_responses=True,
                    encoding="utf-8"
                )
                logger.info("Synchronous Redis connection created successfully")
            except Exception as e:
                logger.error(f"Failed to create synchronous Redis connection: {e}")
                raise
        return self._sync_redis

    async def set_cache(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set a value in Redis cache.
        
        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl: Time to live in seconds (defaults to CACHE_TTL)
            
        Returns:
            True if successful, False otherwise
        """
        if not settings.CACHE_ENABLED:
            return False
            
        try:
            redis_client = await self.get_redis()
            serialized_value = json.dumps(value, default=str)
            ttl = ttl or settings.CACHE_TTL
            
            await redis_client.setex(key, ttl, serialized_value)
            logger.debug(f"Cached value for key: {key} with TTL: {ttl}")
            return True
        except (ConnectionError, RedisError, json.JSONDecodeError) as e:
            logger.error(f"Failed to set cache for key {key}: {e}")
            return False

    async def get_cache(self, key: str) -> Optional[Any]:
        """
        Get a value from Redis cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found, None otherwise
        """
        if not settings.CACHE_ENABLED:
            return None
            
        try:
            redis_client = await self.get_redis()
            cached_value = await redis_client.get(key)
            
            if cached_value is None:
                logger.debug(f"Cache miss for key: {key}")
                return None
                
            deserialized_value = json.loads(cached_value)
            logger.debug(f"Cache hit for key: {key}")
            return deserialized_value
        except (ConnectionError, RedisError, json.JSONDecodeError) as e:
            logger.error(f"Failed to get cache for key {key}: {e}")
            return None

    async def delete_cache(self, key: str) -> bool:
        """
        Delete a value from Redis cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        if not settings.CACHE_ENABLED:
            return False
            
        try:
            redis_client = await self.get_redis()
            result = await redis_client.delete(key)
            logger.debug(f"Deleted cache for key: {key}")
            return bool(result)
        except (ConnectionError, RedisError) as e:
            logger.error(f"Failed to delete cache for key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in Redis cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists, False otherwise
        """
        if not settings.CACHE_ENABLED:
            return False
            
        try:
            redis_client = await self.get_redis()
            result = await redis_client.exists(key)
            return bool(result)
        except (ConnectionError, RedisError) as e:
            logger.error(f"Failed to check existence for key {key}: {e}")
            return False

    async def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching a pattern.
        
        Args:
            pattern: Redis key pattern (e.g., "weather:*")
            
        Returns:
            Number of keys deleted
        """
        if not settings.CACHE_ENABLED:
            return 0
            
        try:
            redis_client = await self.get_redis()
            keys = await redis_client.keys(pattern)
            if keys:
                deleted = await redis_client.delete(*keys)
                logger.info(f"Cleared {deleted} keys matching pattern: {pattern}")
                return deleted
            return 0
        except (ConnectionError, RedisError) as e:
            logger.error(f"Failed to clear pattern {pattern}: {e}")
            return 0

    async def health_check(self) -> bool:
        """
        Check Redis connection health.
        
        Returns:
            True if Redis is healthy, False otherwise
        """
        try:
            redis_client = await self.get_redis()
            await redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    async def close(self):
        """Close Redis connections."""
        try:
            if self._redis_pool:
                await self._redis_pool.disconnect()
                logger.info("Redis connection pool closed")
            if self._sync_redis:
                await self._sync_redis.aclose()
                logger.info("Synchronous Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connections: {e}")


# Global Redis client instance
redis_client = RedisClient()


def generate_cache_key(*args, **kwargs) -> str:
    """
    Generate a cache key from arguments.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Generated cache key
    """
    key_parts = []
    
    # Add positional arguments
    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            key_parts.append(str(arg))
        else:
            key_parts.append(str(hash(str(arg))))
    
    # Add keyword arguments (sorted for consistency)
    for k, v in sorted(kwargs.items()):
        if isinstance(v, (str, int, float, bool)):
            key_parts.append(f"{k}:{v}")
        else:
            key_parts.append(f"{k}:{hash(str(v))}")
    
    return ":".join(key_parts)
