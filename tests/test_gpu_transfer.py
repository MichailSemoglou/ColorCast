"""Tests for GPU-accelerated transfer functions.

Both the GPU and CPU fallback paths are tested. When CuPy is not available
every function uses its CPU fallback, which is the common path in CI.
"""

import numpy as np
import pytest
from colorcast.processing.gpu_transfer import (
    gpu_histogram_matching,
    gpu_mean_std_transfer,
    gpu_lab_transfer,
    gpu_histogram_matching_multichannel,
    is_gpu_available,
)


class TestIsGpuAvailable:
    """Tests for the GPU availability check."""

    def test_returns_bool(self):
        result = is_gpu_available()
        assert isinstance(result, bool)


class TestGpuHistogramMatching:
    """Tests for GPU-accelerated histogram matching."""

    def test_basic_transfer(self):
        source = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
        reference = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
        result = gpu_histogram_matching(source, reference)
        assert result.shape == source.shape
        assert result.dtype == source.dtype

    def test_output_is_uint8(self):
        source = (np.random.rand(50, 50, 3) * 255).astype(np.uint8)
        reference = (np.random.rand(50, 50, 3) * 255).astype(np.uint8)
        result = gpu_histogram_matching(source, reference)
        assert result.dtype == np.uint8

    def test_identical_images(self):
        image = (np.random.rand(80, 80, 3) * 255).astype(np.uint8)
        result = gpu_histogram_matching(image, image)
        assert result.shape == image.shape

    def test_no_nan_no_inf(self):
        source = (np.random.rand(50, 50, 3) * 255).astype(np.uint8)
        reference = (np.random.rand(50, 50, 3) * 255).astype(np.uint8)
        result = gpu_histogram_matching(source, reference)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestGpuMeanStdTransfer:
    """Tests for GPU-accelerated mean/std transfer."""

    def test_basic_transfer(self):
        source = np.random.rand(100, 100, 3).astype(np.float32)
        reference = np.random.rand(100, 100, 3).astype(np.float32)
        result = gpu_mean_std_transfer(source, reference)
        assert result.shape == source.shape

    def test_output_in_range(self):
        source = np.random.rand(100, 100, 3).astype(np.float32)
        reference = np.random.rand(100, 100, 3).astype(np.float32)
        result = gpu_mean_std_transfer(source, reference)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_identical_images(self):
        image = np.random.rand(80, 80, 3).astype(np.float32)
        result = gpu_mean_std_transfer(image, image)
        np.testing.assert_allclose(result, image, atol=1e-7)

    def test_constant_image(self):
        source = np.full((50, 50, 3), 0.5, dtype=np.float32)
        reference = np.full((50, 50, 3), 0.8, dtype=np.float32)
        result = gpu_mean_std_transfer(source, reference)
        assert result.shape == source.shape
        np.testing.assert_allclose(result, 0.8, atol=1e-5)

    def test_no_nan_no_inf(self):
        source = np.random.rand(50, 50, 3).astype(np.float32)
        reference = np.random.rand(50, 50, 3).astype(np.float32)
        result = gpu_mean_std_transfer(source, reference)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_deterministic(self):
        source = np.random.rand(50, 50, 3).astype(np.float32)
        reference = np.random.rand(50, 50, 3).astype(np.float32)
        r1 = gpu_mean_std_transfer(source, reference)
        r2 = gpu_mean_std_transfer(source, reference)
        np.testing.assert_array_equal(r1, r2)


class TestGpuLabTransfer:
    """Tests for GPU-accelerated Lab color space transfer."""

    def test_basic_transfer(self):
        source = np.random.rand(100, 100, 3).astype(np.float32)
        reference = np.random.rand(100, 100, 3).astype(np.float32)
        result = gpu_lab_transfer(source, reference)
        assert result.shape == source.shape
        assert result.dtype == source.dtype

    def test_output_in_range(self):
        source = np.random.rand(100, 100, 3).astype(np.float32)
        reference = np.random.rand(100, 100, 3).astype(np.float32)
        result = gpu_lab_transfer(source, reference)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    @pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_alpha_parameter(self, alpha):
        source = np.random.rand(50, 50, 3).astype(np.float32)
        reference = np.random.rand(50, 50, 3).astype(np.float32)
        result = gpu_lab_transfer(source, reference, alpha=alpha)
        assert result.shape == source.shape
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

    def test_alpha_zero_returns_source(self):
        source = np.random.rand(50, 50, 3).astype(np.float32)
        reference = np.random.rand(50, 50, 3).astype(np.float32)
        result = gpu_lab_transfer(source, reference, alpha=0.0)
        np.testing.assert_array_almost_equal(result, source, decimal=4)

    def test_alpha_clamped(self):
        source = np.random.rand(50, 50, 3).astype(np.float32)
        reference = np.random.rand(50, 50, 3).astype(np.float32)
        r1 = gpu_lab_transfer(source, reference, alpha=1.5)
        r2 = gpu_lab_transfer(source, reference, alpha=-0.5)
        r3 = gpu_lab_transfer(source, reference, alpha=1.0)
        np.testing.assert_array_almost_equal(r1, r3, decimal=5)
        np.testing.assert_array_almost_equal(r2, source, decimal=4)

    def test_no_nan_no_inf(self):
        source = np.random.rand(50, 50, 3).astype(np.float32)
        reference = np.random.rand(50, 50, 3).astype(np.float32)
        result = gpu_lab_transfer(source, reference)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_identical_images(self):
        image = np.random.rand(50, 50, 3).astype(np.float32)
        result = gpu_lab_transfer(image, image)
        assert result.shape == image.shape
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

    def test_deterministic(self):
        source = np.random.rand(50, 50, 3).astype(np.float32)
        reference = np.random.rand(50, 50, 3).astype(np.float32)
        r1 = gpu_lab_transfer(source, reference, alpha=0.5)
        r2 = gpu_lab_transfer(source, reference, alpha=0.5)
        np.testing.assert_array_equal(r1, r2)


class TestGpuHistogramMatchingMultichannel:
    """Tests for GPU-accelerated multichannel histogram matching."""

    def test_basic_transfer(self):
        source = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
        reference = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
        result = gpu_histogram_matching_multichannel(source, reference)
        assert result.shape == source.shape

    def test_identical_images(self):
        image = (np.random.rand(80, 80, 3) * 255).astype(np.uint8)
        result = gpu_histogram_matching_multichannel(image, image)
        assert result.shape == image.shape
        assert result.dtype == np.float32

    def test_no_crash_on_small_inputs(self):
        source = (np.random.rand(10, 10, 3) * 255).astype(np.uint8)
        reference = (np.random.rand(5, 5, 3) * 255).astype(np.uint8)
        result = gpu_histogram_matching_multichannel(source, reference)
        assert result.shape == source.shape
