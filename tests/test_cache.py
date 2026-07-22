"""Tests for LRU cache functionality."""

import pytest
import numpy as np
from colorcast.processing.cache import StyleTransferCache


class TestStyleTransferCache:
    """Tests for styled image caching."""

    def test_cache_initialization(self):
        """Test cache initialization."""
        cache = StyleTransferCache(max_size=5)

        assert cache.size() == 0
        assert cache.max_size == 5

    def test_cache_get_miss(self):
        """Test cache miss returns None."""
        cache = StyleTransferCache(max_size=5)

        content = np.random.rand(100, 100, 3)
        style = np.random.rand(100, 100, 3)

        result = cache.get(content, style, "histogram")

        assert result is None

    def test_cache_set_and_get(self):
        """Test cache set and get."""
        cache = StyleTransferCache(max_size=5)

        content = np.random.rand(100, 100, 3)
        style = np.random.rand(100, 100, 3)
        styled = np.random.rand(100, 100, 3)

        cache.set(content, style, "histogram", styled)
        result = cache.get(content, style, "histogram")

        assert result is not None
        np.testing.assert_array_equal(result, styled)

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = StyleTransferCache(max_size=3)

        # Add 3 items
        for i in range(3):
            content = np.random.rand(100, 100, 3)
            style = np.random.rand(100, 100, 3)
            styled = np.random.rand(100, 100, 3)
            cache.set(content, style, "histogram", styled)

        assert cache.size() == 3

        # Add 4th item (should evict oldest)
        content4 = np.random.rand(100, 100, 3)
        style4 = np.random.rand(100, 100, 3)
        styled4 = np.random.rand(100, 100, 3)
        cache.set(content4, style4, "histogram", styled4)

        assert cache.size() == 3

    def test_cache_different_methods(self):
        """Test caching with different methods."""
        cache = StyleTransferCache(max_size=5)

        content = np.random.rand(100, 100, 3)
        style = np.random.rand(100, 100, 3)

        styled1 = np.random.rand(100, 100, 3)
        styled2 = np.random.rand(100, 100, 3)

        cache.set(content, style, "histogram", styled1)
        cache.set(content, style, "meanstd", styled2)

        result1 = cache.get(content, style, "histogram")
        result2 = cache.get(content, style, "meanstd")

        np.testing.assert_array_equal(result1, styled1)
        np.testing.assert_array_equal(result2, styled2)

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = StyleTransferCache(max_size=5)

        content = np.random.rand(100, 100, 3)
        style = np.random.rand(100, 100, 3)
        styled = np.random.rand(100, 100, 3)

        cache.set(content, style, "histogram", styled)
        assert cache.size() == 1

        cache.clear()
        assert cache.size() == 0

    def test_cache_with_parameters(self):
        """Test caching with method parameters."""
        cache = StyleTransferCache(max_size=5)

        content = np.random.rand(100, 100, 3)
        style = np.random.rand(100, 100, 3)

        styled1 = np.random.rand(100, 100, 3)
        styled2 = np.random.rand(100, 100, 3)

        cache.set(content, style, "selective_shadows", styled1, params={"shadow_threshold": 0.3})
        cache.set(content, style, "selective_shadows", styled2, params={"shadow_threshold": 0.5})

        result1 = cache.get(content, style, "selective_shadows", params={"shadow_threshold": 0.3})
        result2 = cache.get(content, style, "selective_shadows", params={"shadow_threshold": 0.5})

        np.testing.assert_array_equal(result1, styled1)
        np.testing.assert_array_equal(result2, styled2)

    def test_cache_lru_eviction_order(self):
        """Recently accessed entries survive eviction."""
        cache = StyleTransferCache(max_size=2)

        style = np.zeros((4, 4, 3), dtype=np.float32)
        c1 = np.full((4, 4, 3), 0.1, dtype=np.float32)
        c2 = np.full((4, 4, 3), 0.2, dtype=np.float32)
        c3 = np.full((4, 4, 3), 0.3, dtype=np.float32)

        cache.set(c1, style, "histogram", c1)
        cache.set(c2, style, "histogram", c2)
        cache.get(c1, style, "histogram")  # c1 is now most recently used
        cache.set(c3, style, "histogram", c3)  # evicts c2, not c1

        assert cache.get(c1, style, "histogram") is not None
        assert cache.get(c2, style, "histogram") is None
        assert cache.get(c3, style, "histogram") is not None

    def test_cache_reference_free_method(self):
        """A None style produces a style-independent cache key."""
        cache = StyleTransferCache(max_size=5)

        content = np.random.rand(50, 50, 3)
        styled = np.random.rand(50, 50, 3)

        cache.set(content, None, "simulate_protanopia", styled)
        result = cache.get(content, None, "simulate_protanopia")

        np.testing.assert_array_equal(result, styled)

        # A style-keyed entry for the same content is a different key
        style = np.random.rand(50, 50, 3)
        assert cache.get(content, style, "simulate_protanopia") is None