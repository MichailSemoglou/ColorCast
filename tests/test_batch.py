"""Tests for batch processing functionality."""

from pathlib import Path

import numpy as np
import pytest

from colorcast.processing.batch import BatchProcessor
from colorcast.processing.image_loader import save_image
from colorcast.processing.transfer_methods import match_histograms_multichannel


@pytest.fixture
def sample_image():
    """Create a sample RGB image."""
    return np.random.rand(256, 256, 3).astype(np.float32)


@pytest.fixture
def sample_style_image():
    """Create a sample style image."""
    return np.random.rand(256, 256, 3).astype(np.float32) * 0.5 + 0.25


class TestBatchProcessor:
    """Test BatchProcessor class."""

    def test_initialization(self):
        """Test batch processor initialization."""
        processor = BatchProcessor(
            transfer_method=match_histograms_multichannel,
            max_workers=4,
        )

        assert processor.transfer_method == match_histograms_multichannel
        assert processor.max_workers == 4
        assert processor.progress_callback is None
        assert processor.failed_files == []

    def test_initialization_with_callback(self):
        """Test initialization with progress callback."""
        callback_data = {"count": 0}

        def callback(processed, total):
            callback_data["count"] += 1

        processor = BatchProcessor(
            transfer_method=match_histograms_multichannel,
            max_workers=2,
            progress_callback=callback,
        )

        assert processor.progress_callback == callback

    def test_process_directory_success(self, tmp_path, sample_image, sample_style_image):
        """Test successful directory processing."""

        # Create test directory
        content_dir = tmp_path / "content"
        content_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Save sample images
        for i in range(3):
            save_image(sample_image, str(content_dir / f"test_{i}.jpg"))
        save_image(sample_style_image, str(tmp_path / "style.jpg"))

        # Process batch
        processor = BatchProcessor(
            transfer_method=match_histograms_multichannel,
            max_workers=2,
        )
        results = processor.process_directory(
            content_dir=content_dir,
            style_image=tmp_path / "style.jpg",
            output_dir=output_dir,
            pattern="*.jpg",
        )

        # Verify results
        assert len(results) == 3
        for result_path in results:
            assert result_path.exists()
            assert result_path.suffix == ".jpg"

        # Check no failed files
        assert len(processor.failed_files) == 0

    def test_process_directory_with_callback(self, tmp_path, sample_image, sample_style_image):
        """Test directory processing with progress callback."""

        # Create test directory
        content_dir = tmp_path / "content"
        content_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Save sample images
        for i in range(5):
            save_image(sample_image, str(content_dir / f"test_{i}.jpg"))
        save_image(sample_style_image, str(tmp_path / "style.jpg"))

        # Track callback calls
        callback_calls = []

        def callback(processed, total):
            callback_calls.append((processed, total))

        # Process batch
        processor = BatchProcessor(
            transfer_method=match_histograms_multichannel,
            max_workers=2,
            progress_callback=callback,
        )
        processor.process_directory(
            content_dir=content_dir,
            style_image=tmp_path / "style.jpg",
            output_dir=output_dir,
        )

        # Verify callback was called
        assert len(callback_calls) > 0
        final_call = callback_calls[-1]
        assert final_call[0] == 5  # processed
        assert final_call[1] == 5  # total

    def test_process_directory_pattern_filter(self, tmp_path, sample_image, sample_style_image):
        """Test pattern filtering in directory processing."""

        # Create test directory
        content_dir = tmp_path / "content"
        content_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Save sample images with different extensions
        save_image(sample_image, str(content_dir / "test_1.jpg"))
        save_image(sample_image, str(content_dir / "test_2.png"))
        save_image(sample_image, str(content_dir / "test_3.jpg"))
        save_image(sample_style_image, str(tmp_path / "style.jpg"))

        # Process only JPG files
        processor = BatchProcessor(
            transfer_method=match_histograms_multichannel,
        )
        results = processor.process_directory(
            content_dir=content_dir,
            style_image=tmp_path / "style.jpg",
            output_dir=output_dir,
            pattern="*.jpg",
        )

        # Should only process 2 JPG files
        assert len(results) == 2

    def test_process_directory_error_handling(self, tmp_path, sample_image, sample_style_image):
        """Test error handling in directory processing."""

        # Create test directory
        content_dir = tmp_path / "content"
        content_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Save valid and invalid images
        save_image(sample_image, str(content_dir / "valid_1.jpg"))
        save_image(sample_image, str(content_dir / "valid_2.jpg"))
        # Create invalid JPEG file (corrupted data)
        (content_dir / "invalid.jpg").write_bytes(b"not a valid jpeg file")
        save_image(sample_style_image, str(tmp_path / "style.jpg"))

        # Process batch
        processor = BatchProcessor(
            transfer_method=match_histograms_multichannel,
        )
        results = processor.process_directory(
            content_dir=content_dir,
            style_image=tmp_path / "style.jpg",
            output_dir=output_dir,
        )

        # Should process 2 valid files
        assert len(results) == 2

        # Should have 1 failed file (the corrupted one)
        assert len(processor.failed_files) == 1
        assert "invalid.jpg" in str(processor.failed_files[0][0])

    def test_failed_files_tracking(self, tmp_path, sample_image, sample_style_image):
        """Test that failed files are properly tracked."""

        # Create test directory
        content_dir = tmp_path / "content"
        content_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Save valid and create invalid JPEG file
        save_image(sample_image, str(content_dir / "valid.jpg"))
        # Create corrupted JPEG file
        (content_dir / "invalid.jpg").write_bytes(b"corrupted jpeg data")
        save_image(sample_style_image, str(tmp_path / "style.jpg"))

        # Process batch
        processor = BatchProcessor(
            transfer_method=match_histograms_multichannel,
        )
        processor.process_directory(
            content_dir=content_dir,
            style_image=tmp_path / "style.jpg",
            output_dir=output_dir,
        )

        # Check failed files are tracked
        assert len(processor.failed_files) == 1
        failed_path, error_msg = processor.failed_files[0]
        assert isinstance(failed_path, Path)
        assert "invalid.jpg" in str(failed_path)
        assert isinstance(error_msg, str)
        assert len(error_msg) > 0
