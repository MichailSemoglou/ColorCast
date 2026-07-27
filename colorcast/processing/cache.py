"""LRU cache for styled images."""

import hashlib
import threading
from collections import OrderedDict
from typing import Callable, Optional
import numpy as np

#: Key component used for methods that do not take a reference (style) image.
_NO_REFERENCE = "no-reference"


class StyleTransferCache:
    """LRU cache for styled images with method-aware storage.

    Backed by :class:`collections.OrderedDict`: an entry is moved to the end
    on every access and the front entry is evicted when the cache is full,
    so no hand-ordered bookkeeping is needed.
    """

    def __init__(self, max_size: int = 8):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of styled images to cache
        """
        self.max_size = max_size
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

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
            style_hash: Hash of style image, or a marker for reference-free
                methods
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

        Hashes a coarse fingerprint (shape, dtype, strides, and a
        downsampled view of the pixel data) rather than the full image
        bytes. The collision risk is negligible for the cache sizes this
        module targets.

        Args:
            img: Image array to hash

        Returns:
            First 16 characters of MD5 hash
        """
        stride = max(1, min(img.shape[0], img.shape[1]) // 64)
        subsample = img[::stride, ::stride]
        h = hashlib.md5()
        h.update(str(img.shape).encode())
        h.update(str(img.dtype).encode())
        h.update(str(img.strides).encode())
        h.update(subsample.tobytes())
        return h.hexdigest()[:16]

    def _style_hash(self, style: Optional[np.ndarray]) -> str:
        """Hash the style image, or return the reference-free marker."""
        if style is None:
            return _NO_REFERENCE
        return self._compute_hash(style)

    def get(
        self,
        content: np.ndarray,
        style: Optional[np.ndarray],
        method: str,
        params: Optional[dict] = None,
    ) -> Optional[np.ndarray]:
        """
        Retrieve cached styled image.

        Args:
            content: Content image array
            style: Style image array, or None for reference-free methods
            method: Transfer method ID
            params: Method parameters

        Returns:
            Cached styled image if found, None otherwise
        """
        if params is None:
            params = {}

        key = self._generate_key(
            self._compute_hash(content),
            self._style_hash(style),
            method,
            params,
        )

        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]

            self.misses += 1
            return None

    def set(
        self,
        content: np.ndarray,
        style: Optional[np.ndarray],
        method: str,
        styled: np.ndarray,
        params: Optional[dict] = None,
    ) -> None:
        """
        Cache styled image with LRU eviction.

        Args:
            content: Content image array
            style: Style image array, or None for reference-free methods
            method: Transfer method ID
            styled: Styled image to cache
            params: Method parameters
        """
        if params is None:
            params = {}

        key = self._generate_key(
            self._compute_hash(content),
            self._style_hash(style),
            method,
            params,
        )

        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            elif len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)

            self.cache[key] = styled

    def clear(self) -> None:
        """Clear all cached images."""
        with self._lock:
            self.cache.clear()

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
        compute_func: Callable[[], np.ndarray],
    ) -> np.ndarray:
        """
        Get value from cache or compute and cache it.

        Args:
            key: Cache key
            compute_func: Function to compute value if not in cache

        Returns:
            Cached or computed value
        """
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]

        value: np.ndarray = compute_func()

        with self._lock:
            self.misses += 1
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]

            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)

            self.cache[key] = value
            return value


# Alias for backward compatibility
LRUCache = StyleTransferCache
