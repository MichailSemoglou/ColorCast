"""LRU cache for styled images."""

import hashlib
from typing import Optional, Dict, Any
import numpy as np


class StyleTransferCache:
    """LRU cache for styled images with method-aware storage."""

    def __init__(self, max_size: int = 8):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of styled images to cache
        """
        self.max_size = max_size
        self.cache: Dict[str, np.ndarray] = {}
        self.access_order: list = []
        self.hits = 0
        self.misses = 0

    def _generate_key(
        self,
        content_hash: str,
        style_hash: str,
        method: str,
        params: dict,
    ) -> str:
        """
        Generate cache key from parameters.

        Args:
            content_hash: Hash of content image
            style_hash: Hash of style image
            method: Transfer method ID
            params: Method parameters

        Returns:
            Cache key string
        """
        params_str = str(sorted(params.items()))
        return f"{content_hash}:{style_hash}:{method}:{params_str}"

    def _compute_hash(self, img: np.ndarray) -> str:
        """
        Compute hash of image array.

        Args:
            img: Image array to hash

        Returns:
            First 16 characters of MD5 hash
        """
        return hashlib.md5(img.tobytes()).hexdigest()[:16]

    def get(
        self,
        content: np.ndarray,
        style: np.ndarray,
        method: str,
        params: dict = None,
    ) -> Optional[np.ndarray]:
        """
        Retrieve cached styled image.

        Args:
            content: Content image array
            style: Style image array
            method: Transfer method ID
            params: Method parameters

        Returns:
            Cached styled image if found, None otherwise
        """
        if params is None:
            params = {}

        key = self._generate_key(
            self._compute_hash(content),
            self._compute_hash(style),
            method,
            params,
        )

        if key in self.cache:
            # Update access order (LRU)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]

        return None

    def set(
        self,
        content: np.ndarray,
        style: np.ndarray,
        method: str,
        styled: np.ndarray,
        params: dict = None,
    ) -> None:
        """
        Cache styled image with LRU eviction.

        Args:
            content: Content image array
            style: Style image array
            method: Transfer method ID
            styled: Styled image to cache
            params: Method parameters
        """
        if params is None:
            params = {}

        # Evict oldest entry if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]

        key = self._generate_key(
            self._compute_hash(content),
            self._compute_hash(style),
            method,
            params,
        )

        self.cache[key] = styled
        self.access_order.append(key)

    def clear(self) -> None:
        """Clear all cached images."""
        self.cache.clear()
        self.access_order.clear()

    def size(self) -> int:
        """
        Get current cache size.

        Returns:
            Number of cached images
        """
        return len(self.cache)

    def stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics (hits, misses, size)
        """
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": len(self.cache),
        }

    def __contains__(self, key: str) -> bool:
        """Check if key is in cache."""
        return key in self.cache

    def get_or_compute(
        self,
        key: str,
        compute_func: callable,
    ) -> np.ndarray:
        """
        Get value from cache or compute and cache it.

        Args:
            key: Cache key
            compute_func: Function to compute value if not in cache

        Returns:
            Cached or computed value
        """
        if key in self.cache:
            # Update access order (LRU)
            self.access_order.remove(key)
            self.access_order.append(key)
            self.hits += 1
            return self.cache[key]
        
        # Compute value
        value = compute_func()
        self.misses += 1
        
        # Evict oldest entry if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        
        # Cache the computed value
        self.cache[key] = value
        self.access_order.append(key)
        
        return value


# Alias for backward compatibility
LRUCache = StyleTransferCache
