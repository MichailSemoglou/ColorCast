"""Tests for visualization functions."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from colorcast.analysis.visualization import (
    create_side_by_side_comparison,
    visualize_color_channels,
    visualize_method_comparison,
    visualize_transfer_result,
)


@pytest.fixture(autouse=True)
def close_figures():
    """Close all Matplotlib figures after each test to prevent resource leaks."""
    yield
    plt.close("all")


class TestVisualizeTransferResult:
    """Tests for visualize_transfer_result."""

    @pytest.fixture
    def images(self):
        return (
            np.random.rand(100, 100, 3).astype(np.float32),
            np.random.rand(100, 100, 3).astype(np.float32),
            np.random.rand(100, 100, 3).astype(np.float32),
        )

    def test_returns_figure(self, images):
        source, reference, result = images
        fig = visualize_transfer_result(source, reference, result, "Test Method")
        assert isinstance(fig, Figure)

    def test_default_no_histograms_no_difference(self, images):
        source, reference, result = images
        fig = visualize_transfer_result(
            source,
            reference,
            result,
            "Test",
            show_histograms=False,
            show_difference=False,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_default_with_difference(self, images):
        source, reference, result = images
        fig = visualize_transfer_result(
            source,
            reference,
            result,
            "Test",
            show_histograms=False,
            show_difference=True,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_with_histograms_no_difference(self, images):
        source, reference, result = images
        fig = visualize_transfer_result(
            source,
            reference,
            result,
            "Test",
            show_histograms=True,
            show_difference=False,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_with_histograms_and_difference(self, images):
        source, reference, result = images
        fig = visualize_transfer_result(
            source,
            reference,
            result,
            "Test",
            show_histograms=True,
            show_difference=True,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_figsize(self, images):
        source, reference, result = images
        fig = visualize_transfer_result(
            source,
            reference,
            result,
            "Test",
            figsize=(10, 8),
        )
        assert isinstance(fig, Figure)

    @pytest.mark.parametrize("show_histograms", [True, False])
    @pytest.mark.parametrize("show_difference", [True, False])
    def test_all_combinations(self, images, show_histograms, show_difference):
        source, reference, result = images
        fig = visualize_transfer_result(
            source,
            reference,
            result,
            "Test",
            show_histograms=show_histograms,
            show_difference=show_difference,
        )
        assert isinstance(fig, Figure)


class TestVisualizeMethodComparison:
    """Tests for visualize_method_comparison."""

    @pytest.fixture
    def reference(self):
        return np.random.rand(80, 80, 3).astype(np.float32)

    def test_returns_figure(self, reference):
        images = {"Method A": np.random.rand(80, 80, 3).astype(np.float32)}
        fig = visualize_method_comparison(images, reference)
        assert isinstance(fig, Figure)

    def test_multiple_methods(self, reference):
        images = {
            "A": np.random.rand(80, 80, 3).astype(np.float32),
            "B": np.random.rand(80, 80, 3).astype(np.float32),
            "C": np.random.rand(80, 80, 3).astype(np.float32),
        }
        fig = visualize_method_comparison(images, reference)
        assert isinstance(fig, Figure)

    def test_no_difference(self, reference):
        images = {"A": np.random.rand(80, 80, 3).astype(np.float32)}
        fig = visualize_method_comparison(images, reference, show_difference=False)
        assert isinstance(fig, Figure)

    def test_custom_figsize(self, reference):
        images = {"A": np.random.rand(80, 80, 3).astype(np.float32)}
        fig = visualize_method_comparison(images, reference, figsize=(15, 10))
        assert isinstance(fig, Figure)

    def test_single_method(self, reference):
        images = {"A": np.random.rand(80, 80, 3).astype(np.float32)}
        fig = visualize_method_comparison(images, reference, show_difference=False)
        assert isinstance(fig, Figure)


class TestVisualizeColorChannels:
    """Tests for visualize_color_channels."""

    def test_returns_figure(self):
        image = np.random.rand(100, 100, 3).astype(np.float32)
        fig = visualize_color_channels(image)
        assert isinstance(fig, Figure)

    def test_custom_title(self):
        image = np.random.rand(100, 100, 3).astype(np.float32)
        fig = visualize_color_channels(image, title="Custom Title")
        assert isinstance(fig, Figure)

    def test_custom_figsize(self):
        image = np.random.rand(100, 100, 3).astype(np.float32)
        fig = visualize_color_channels(image, figsize=(8, 2))
        assert isinstance(fig, Figure)


class TestCreateSideBySideComparison:
    """Tests for create_side_by_side_comparison."""

    def test_returns_figure(self):
        img1 = np.random.rand(100, 100, 3).astype(np.float32)
        img2 = np.random.rand(100, 100, 3).astype(np.float32)
        fig = create_side_by_side_comparison(img1, img2)
        assert isinstance(fig, Figure)

    def test_custom_labels(self):
        img1 = np.random.rand(100, 100, 3).astype(np.float32)
        img2 = np.random.rand(100, 100, 3).astype(np.float32)
        fig = create_side_by_side_comparison(
            img1,
            img2,
            label1="Original",
            label2="Styled",
        )
        assert isinstance(fig, Figure)

    def test_custom_figsize(self):
        img1 = np.random.rand(100, 100, 3).astype(np.float32)
        img2 = np.random.rand(100, 100, 3).astype(np.float32)
        fig = create_side_by_side_comparison(img1, img2, figsize=(8, 4))
        assert isinstance(fig, Figure)
