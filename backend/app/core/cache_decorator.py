"""
Caching decorators for API calls and functions.

This module provides decorators to cache function results using Redis.
"""

import asyncio
import functools
import logging
from typing import Callable, Optional

from app.core.redis_client import redis_client, generate_cache_key

logger = logging.getLogger(__name__)


def cache_result(
    ttl: Optional[int] = None,
    key_prefix: str = "",
    ignore_args: Optional[list] = None,
    ignore_kwargs: Optional[list] = None
):
    """
    Decorator to cache function results in Redis.
    
    Args:
        ttl: Time to live in seconds (defaults to CACHE_TTL from settings)
        key_prefix: Prefix for cache keys
        ignore_args: List of argument indices to ignore when generating cache key
        ignore_kwargs: List of keyword argument names to ignore when generating cache key
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Filter out ignored arguments
            filtered_args = args
            filtered_kwargs = kwargs.copy()
            
            if ignore_args:
                filtered_args = tuple(arg for i, arg in enumerate(args) if i not in ignore_args)
            
            if ignore_kwargs:
                for key in ignore_kwargs:
                    filtered_kwargs.pop(key, None)
            
            # Generate cache key
            cache_key = f"{key_prefix}:{func.__name__}:{generate_cache_key(*filtered_args, **filtered_kwargs)}"
            
            # Try to get from cache first
            cached_result = await redis_client.get_cache(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for function {func.__name__}")
                return cached_result
            
            # Execute function if not in cache
            logger.debug(f"Cache miss for function {func.__name__}, executing...")
            
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Cache the result
            await redis_client.set_cache(cache_key, result, ttl)
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we need to handle async cache operations
            loop = None
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            return loop.run_until_complete(async_wrapper(*args, **kwargs))
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def cache_api_call(
    ttl: int = 3600,
    key_prefix: str = "api"
):
    """
    Decorator specifically for caching external API calls.
    
    Args:
        ttl: Time to live in seconds (default 1 hour)
        key_prefix: Prefix for cache keys
    """
    return cache_result(ttl=ttl, key_prefix=key_prefix)


def invalidate_cache_pattern(pattern: str):
    """
    Decorator to invalidate cache entries matching a pattern after function execution.
    
    Args:
        pattern: Redis key pattern to invalidate
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Execute function first
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Invalidate cache pattern
            await redis_client.clear_pattern(pattern)
            logger.info(f"Invalidated cache pattern: {pattern}")
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            loop = None
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            return loop.run_until_complete(async_wrapper(*args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class CacheManager:
    """Cache management utilities."""
    
    @staticmethod
    async def warm_cache(func: Callable, *args, **kwargs):
        """
        Pre-warm cache by executing function and storing result.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
        """
        try:
            if asyncio.iscoroutinefunction(func):
                await func(*args, **kwargs)
            else:
                func(*args, **kwargs)
            logger.info(f"Cache warmed for function: {func.__name__}")
        except Exception as e:
            logger.error(f"Failed to warm cache for {func.__name__}: {e}")
    
    @staticmethod
    async def clear_function_cache(func: Callable, key_prefix: str = ""):
        """
        Clear all cache entries for a specific function.
        
        Args:
            func: Function whose cache to clear
            key_prefix: Key prefix used when caching
        """
        pattern = f"{key_prefix}:{func.__name__}:*"
        cleared = await redis_client.clear_pattern(pattern)
        logger.info(f"Cleared {cleared} cache entries for function: {func.__name__}")
        return cleared
    
    @staticmethod
    async def get_cache_stats() -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            redis_conn = await redis_client.get_redis()
            info = await redis_conn.info()
            
            return {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}


# Global cache manager instance
cache_manager = CacheManager()
