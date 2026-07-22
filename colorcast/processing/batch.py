"""Batch processing utilities for ColorCast."""

import logging
from pathlib import Path
from typing import List, Callable, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from colorcast.processing.image_loader import load_image, save_image
from colorcast.utils.exceptions import ImageLoadError

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Process multiple images in batch with error handling and logging."""

    def __init__(
        self,
        transfer_method: Callable,
        max_workers: int = 4,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        """
        Initialize batch processor.

        Args:
            transfer_method: Function to apply transfer
            max_workers: Number of parallel workers (default: 4)
            progress_callback: Optional callback(int processed, int total)
        """
        self.transfer_method = transfer_method
        self.max_workers = max_workers
        self.progress_callback = progress_callback
        self.failed_files: List[Tuple[Path, str]] = []

    def process_directory(
        self,
        content_dir: Path,
        style_image: Optional[Path],
        output_dir: Path,
        pattern: str = "*.jpg",
    ) -> List[Path]:
        """
        Process all images in directory with same style.

        Args:
            content_dir: Directory with content images
            style_image: Path to style image, or None for methods that do
                not require a reference image
            output_dir: Directory for results
            pattern: File pattern to match

        Returns:
            List of output file paths
        """
        content_dir = Path(content_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load style image once, if provided
        style = load_image(str(style_image)) if style_image is not None else None

        # Find all content images
        content_files = list(content_dir.glob(pattern))
        total = len(content_files)

        def process_single(
            content_path: Path,
        ) -> Tuple[Optional[Path], Optional[Tuple[Path, str]]]:
            """Process single image with error handling.

            Returns:
                ``(output_path, None)`` on success, or
                ``(None, (content_path, error_message))`` on failure. Each
                worker returns its own outcome, so no state shared between
                threads is mutated here.
            """
            try:
                content = load_image(str(content_path))
                result = self.transfer_method(content, style)

                output_path = output_dir / f"styled_{content_path.name}"
                save_image(result, str(output_path))

                logger.info(f"Successfully processed: {content_path.name}")

                return output_path, None
            except ImageLoadError as e:
                error_msg = f"Image load error: {e!s}"
                logger.error(f"Failed to process {content_path.name}: {error_msg}")
                return None, (content_path, error_msg)
            except Exception as e:  # noqa: BLE001
                # Broad on purpose: each worker isolates its own failure so
                # one bad image cannot abort the rest of the batch.
                error_msg = f"Unexpected error: {e!s}"
                logger.error(f"Failed to process {content_path.name}: {error_msg}")
                return None, (content_path, error_msg)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            all_results = list(executor.map(process_single, content_files))

        # Merge per-worker outcomes, keeping successes and failures apart
        results = [path for path, _ in all_results if path is not None]
        failed_files = [failure for _, failure in all_results if failure is not None]

        # Store failed files for reporting
        self.failed_files = failed_files

        # Log summary
        success_count = len(results)
        failure_count = len(failed_files)
        logger.info(
            f"Batch processing complete: {success_count} successful, "
            f"{failure_count} failed out of {total} total files"
        )

        # Update progress callback with final count
        if self.progress_callback:
            self.progress_callback(success_count, total)

        return results

    def process_pairs(
        self,
        image_pairs: List[tuple],
        output_dir: Path,
    ) -> List[Path]:
        """
        Process content/style image pairs.

        Args:
            image_pairs: List of (content_path, style_path) tuples
            output_dir: Directory for results

        Returns:
            List of output file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        failed_pairs = []

        for i, (content_path, style_path) in enumerate(image_pairs):
            try:
                content = load_image(str(content_path))
                style = load_image(str(style_path))

                result = self.transfer_method(content, style)

                output_path = output_dir / f"result_{i:04d}.jpg"
                save_image(result, str(output_path))

                logger.info(
                    f"Successfully processed pair {i+1}: "
                    f"{content_path.name} + {style_path.name}"
                )

                results.append(output_path)
            except ImageLoadError as e:
                error_msg = f"Image load error: {e!s}"
                logger.error(
                    f"Failed to process pair {i+1} "
                    f"({content_path.name}, {style_path.name}): {error_msg}"
                )
                failed_pairs.append(((content_path, style_path), error_msg))
            except Exception as e:  # noqa: BLE001
                # Broad on purpose: each pair is isolated so one bad image
                # cannot abort the rest of the batch.
                error_msg = f"Unexpected error: {e!s}"
                logger.error(
                    f"Failed to process pair {i+1} "
                    f"({content_path.name}, {style_path.name}): {error_msg}"
                )
                failed_pairs.append(((content_path, style_path), error_msg))

            if self.progress_callback:
                self.progress_callback(i + 1, len(image_pairs))

        # Log summary
        success_count = len(results)
        failure_count = len(failed_pairs)
        logger.info(
            f"Pair processing complete: {success_count} successful, "
            f"{failure_count} failed out of {len(image_pairs)} total pairs"
        )

        # Store failed pairs for reporting
        self.failed_files = failed_pairs

        return results
