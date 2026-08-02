"""Tests for LRU cache functionality."""

import threading

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
        for _i in range(3):
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

    def test_cache_clear_resets_stats(self):
        """Test cache clearing resets stored entries and counters."""
        cache = StyleTransferCache(max_size=5)

        content = np.random.rand(50, 50, 3)
        style = np.random.rand(50, 50, 3)
        styled = np.random.rand(50, 50, 3)

        cache.set(content, style, "histogram", styled)
        cache.get(content, style, "histogram")
        cache.get(content, style, "meanstd")

        assert cache.size() == 1
        assert cache.stats()["hits"] == 1
        assert cache.stats()["misses"] == 1

        cache.clear()

        assert cache.size() == 0
        assert cache.stats()["hits"] == 0
        assert cache.stats()["misses"] == 0

    def test_generate_key_matches_internal_derivation(self):
        """The public helper should mirror the internal key derivation."""
        cache = StyleTransferCache(max_size=5)

        content = np.zeros((4, 4, 3), dtype=np.float32)
        style = np.ones((4, 4, 3), dtype=np.float32)
        params = {"shadow_threshold": 0.3}

        expected = cache._generate_key(
            cache._compute_hash(content),
            cache._style_hash(style),
            "histogram",
            params,
        )
        expected_without_params = cache._generate_key(
            cache._compute_hash(content),
            cache._style_hash(style),
            "histogram",
            {},
        )

        assert cache.generate_key(content, style, "histogram", params) == expected
        assert cache.generate_key(content, style, "histogram") == expected_without_params

    def test_generate_key_changes_for_any_pixel_change(self):
        """Any pixel change should produce a different cache key."""
        cache = StyleTransferCache(max_size=5)

        style = np.zeros((256, 256, 3), dtype=np.float32)
        content = np.zeros((256, 256, 3), dtype=np.float32)
        modified_content = content.copy()
        modified_content[5, 5, :] = 1.0

        assert cache.generate_key(content, style, "histogram") != cache.generate_key(
            modified_content,
            style,
            "histogram",
        )

    def test_compute_hash_handles_contiguous_sliced_and_transposed_arrays(self):
        """Hashing should work for contiguous, sliced, and transposed arrays."""
        cache = StyleTransferCache(max_size=5)
        base = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

        contiguous_hash = cache._compute_hash(base)
        sliced_hash = cache._compute_hash(base[:, :2, :])
        transposed_hash = cache._compute_hash(np.transpose(base, (1, 0, 2)))

        assert contiguous_hash
        assert sliced_hash
        assert transposed_hash
        assert contiguous_hash != sliced_hash
        assert contiguous_hash != transposed_hash

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

    def test_get_or_compute_concurrent_same_key(self):
        """Second caller returns the cached result from the first computation.

        Regression test for the double-check inside get_or_compute: when two
        threads both miss the first lock check and the first thread stores its
        result before the second completes, the second thread must return the
        already-cached value and the cache must contain exactly one entry.
        """
        cache = StyleTransferCache(max_size=8)
        content = np.zeros((4, 4, 3), dtype=np.float32)
        style = np.zeros((4, 4, 3), dtype=np.float32)
        key = cache._generate_key(
            cache._compute_hash(content),
            cache._style_hash(style),
            "histogram",
            {},
        )

        first_value = np.full((4, 4, 3), 0.1, dtype=np.float32)
        second_value = np.full((4, 4, 3), 0.9, dtype=np.float32)

        # Thread 1 blocks in its compute_func until thread 2 has also passed
        # the first lock check, so both callers miss the initial lookup.
        # Thread 2 then blocks until thread 1 has stored its result, ensuring
        # the second lock check in thread 2 sees the populated cache.
        second_is_computing = threading.Event()
        first_has_stored = threading.Event()
        results = {}

        def compute_first():
            second_is_computing.wait(timeout=5.0)
            return first_value

        def compute_second():
            second_is_computing.set()
            first_has_stored.wait(timeout=5.0)
            return second_value

        def thread1_fn():
            results["first"] = cache.get_or_compute(key, compute_first)
            first_has_stored.set()

        def thread2_fn():
            results["second"] = cache.get_or_compute(key, compute_second)

        t1 = threading.Thread(target=thread1_fn)
        t2 = threading.Thread(target=thread2_fn)
        t1.start()
        t2.start()
        t1.join(timeout=10.0)
        t2.join(timeout=10.0)

        assert not t1.is_alive(), "thread 1 did not complete"
        assert not t2.is_alive(), "thread 2 did not complete"
        assert cache.size() == 1
        np.testing.assert_array_equal(results["second"], first_value)
