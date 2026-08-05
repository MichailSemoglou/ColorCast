"""Command-line interface for ColorCast."""

import argparse
import sys
from pathlib import Path

import numpy as np

from colorcast import (
    blend_images,
    load_image,
    registry,
    save_image,
)
from colorcast.processing.batch import BatchProcessor
from colorcast.processing.image_loader import ALLOWED_IMAGE_EXTENSIONS
from colorcast.utils.exceptions import ImageProcessingError, ValidationError


def _validate_intensity(value: str) -> float:
    """argparse type validator for --intensity (float in [0, 1])."""
    from colorcast.utils.validators_enhanced import validate_float_parameter

    try:
        f = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"invalid float value: {value!r}") from None
    try:
        return validate_float_parameter(f, "intensity", 0.0, 1.0)
    except ValidationError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="colorcast",
        description="Advanced color and style transfer toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic histogram matching
  colorcast transfer content.jpg style.jpg -o output.jpg

  # Mean/Std transfer with 70% intensity
  colorcast transfer content.jpg style.jpg -o output.jpg -m meanstd -i 0.7

  # Batch process directory
  colorcast batch ./content_dir style.jpg -o ./output_dir

  # List available methods
  colorcast list-methods

  # Selective transfer on shadows
  colorcast transfer content.jpg style.jpg -o output.jpg -m selective_shadows
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --verbose is available on the top-level parser and every subcommand,
    # so that both `colorcast --verbose transfer ...` and
    # `colorcast transfer --verbose ...` work.
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show traceback on errors",
    )

    # Transfer command
    transfer_parser = subparsers.add_parser(
        "transfer",
        help="Apply color transfer between two images",
    )
    transfer_parser.add_argument(
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Show traceback on errors",
    )
    transfer_parser.add_argument(
        "content",
        type=Path,
        help="Path to content image",
    )
    transfer_parser.add_argument(
        "style",
        type=Path,
        nargs="?",
        help="Path to style image (not required for simulate_* methods)",
    )
    transfer_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Path to save result",
    )
    transfer_parser.add_argument(
        "-m",
        "--method",
        default="histogram",
        choices=list(registry.list_methods().keys()),
        help="Transfer method (default: histogram)",
    )
    transfer_parser.add_argument(
        "-i",
        "--intensity",
        type=_validate_intensity,
        default=1.0,
        help="Blend intensity 0.0-1.0 (default: 1.0)",
    )
    transfer_parser.add_argument(
        "--shadow-threshold",
        type=float,
        default=0.3,
        help="Shadow threshold for selective methods (default: 0.3)",
    )
    transfer_parser.add_argument(
        "--highlight-threshold",
        type=float,
        default=0.7,
        help="Highlight threshold for selective methods (default: 0.7)",
    )

    # Batch command
    batch_parser = subparsers.add_parser(
        "batch",
        help="Batch process multiple images",
    )
    batch_parser.add_argument(
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Show traceback on errors",
    )
    batch_parser.add_argument(
        "content_dir",
        type=Path,
        help="Directory containing content images",
    )
    batch_parser.add_argument(
        "style",
        type=Path,
        nargs="?",
        help="Path to style image (not required for simulate_* methods)",
    )
    batch_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output directory for results",
    )
    batch_parser.add_argument(
        "-m",
        "--method",
        default="histogram",
        choices=list(registry.list_methods().keys()),
        help="Transfer method (default: histogram)",
    )
    batch_parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    batch_parser.add_argument(
        "-p",
        "--pattern",
        default="*.jpg",
        help="File pattern to match (default: *.jpg)",
    )

    # List methods command
    subparsers.add_parser(
        "list-methods",
        help="List available transfer methods",
    )

    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show package information",
    )
    info_parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information",
    )

    # Dashboard command
    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Generate a CVD accessibility dashboard report",
    )
    dashboard_parser.add_argument(
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Show traceback on errors",
    )
    dashboard_parser.add_argument(
        "image",
        type=Path,
        help="Path to the image to analyze",
    )
    dashboard_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("dashboard_report.png"),
        help="Output path for the dashboard report (default: dashboard_report.png)",
    )
    dashboard_parser.add_argument(
        "--appearance",
        choices=["cielab", "ictcp"],
        default="cielab",
        help="Appearance space for ΔE ranking (default: cielab)",
    )

    return parser.parse_args()


def _ensure_style_image(method_id: str, style_arg: Path | None, requires_reference: bool) -> None:
    """Raise ValueError if a style image is required but not provided."""
    if requires_reference and style_arg is None:
        raise ValueError(
            f"Method '{method_id}' requires a style image. "
            "Pass one as the second positional argument."
        )


def cmd_transfer(args: argparse.Namespace) -> None:
    """Handle transfer command."""
    method = registry.get_method(args.method)

    print(f"Loading content image: {args.content}")
    content = load_image(str(args.content))

    if method.requires_reference:
        _ensure_style_image(args.method, args.style, True)
        print(f"Loading style image: {args.style}")
        style: np.ndarray | None = load_image(str(args.style))
    else:
        # Simulator methods transform the content image alone.
        style = None

    print(f"Applying {args.method} transfer...")

    # Build kwargs from the subset of CLI args that the method declares.
    # This prevents passing unrelated parameters to reference-free methods
    # (simulators, Daltonizers) whose transfer() signatures would break if
    # they ever stopped accepting **kwargs blindly.
    #
    # Intensity is handled by the post-transfer blend step below, not
    # forwarded as a method parameter.
    _param_map = {
        "shadow_threshold": args.shadow_threshold,
        "highlight_threshold": args.highlight_threshold,
    }
    kwargs = {k: v for k, v in _param_map.items() if k in method.parameters}

    result = method.transfer(content, style, **kwargs)

    # Apply intensity blending
    if args.intensity < 1.0:
        print(f"Blending with intensity: {args.intensity:.2f}")
        result = blend_images(content, result, args.intensity)

    # Save result
    print(f"Saving result to: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(result, str(args.output))

    print("Transfer complete!")


def cmd_batch(args: argparse.Namespace) -> None:
    """Handle batch command."""
    print("Starting batch processing...")
    print(f"Content directory: {args.content_dir}")
    print(f"Output directory: {args.output}")
    print(f"Method: {args.method}")
    print(f"Workers: {args.workers}")

    method = registry.get_method(args.method)

    if method.requires_reference:
        _ensure_style_image(args.method, args.style, True)
        style_image: Path | None = args.style
    else:
        style_image = None

    # Progress callback
    def progress_callback(processed: int, total: int) -> None:
        percent = (processed / total) * 100
        print(f"Progress: {processed}/{total} ({percent:.1f}%)")

    # Process batch
    processor = BatchProcessor(
        transfer_method=method.transfer,
        max_workers=args.workers,
        progress_callback=progress_callback,
    )

    results = processor.process_directory(
        content_dir=args.content_dir,
        style_image=style_image,
        output_dir=args.output,
        pattern=args.pattern,
    )

    # Report results
    print("\nBatch processing complete!")
    print(f"Successfully processed: {len(results)} files")
    if processor.failed_files:
        print(f"Failed: {len(processor.failed_files)} files")
        for failed_path, error in processor.failed_files[:5]:
            print(f"  - {failed_path.name}: {error}")
        if len(processor.failed_files) > 5:
            print(f"  ... and {len(processor.failed_files) - 5} more")


def cmd_list_methods() -> None:
    """Handle list-methods command."""
    print("Available transfer methods:")
    print("-" * 50)

    methods = registry.list_methods()
    for method_id, method_name in sorted(methods.items()):
        print(f"  {method_id:25} - {method_name}")

    print("-" * 50)
    print(f"Total: {len(methods)} methods")


def cmd_info(args: argparse.Namespace) -> None:
    """Handle info command."""
    from colorcast import __author__, __license__, __version__

    print("ColorCast — color and style transfer between images.")
    print("-" * 50)

    if not args.version:
        print("ColorCast provides style transfer, CVD simulation,")
        print("Daltonization correction, and analysis tools.")
        print()

    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print(f"License: {__license__}")

    if not args.version:
        print()
        print("Supported image formats:")
        for ext in ALLOWED_IMAGE_EXTENSIONS:
            print(f"  {ext}")
        print()
        print("Use 'colorcast --help' for usage information.")
        print("Visit https://github.com/MichailSemoglou/ColorCast for more info.")


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Handle dashboard command."""
    from colorcast.analysis.appearance import make_appearance_space
    from colorcast.analysis.dashboard import compute_dashboard, generate_dashboard_report

    image = load_image(str(args.image))
    appearance = make_appearance_space(args.appearance)

    print(f"Computing CVD dashboard (appearance space: {appearance.name})…")
    result = compute_dashboard(image, appearance=appearance)

    print("Summary:")
    for deficiency in ("protanopia", "deuteranopia", "tritanopia"):
        stats = result.summary[deficiency]
        print(
            f"  {deficiency:14}  mean={stats['mean_error']:7.2f}  "
            f"median={stats['median_error']:7.2f}  "
            f"p95={stats['p95_error']:7.2f}  "
            f"affected={stats['percent_affected']:5.1f}%"
        )

    metric_label = result.metric_label
    title = f"CVD Accessibility Dashboard – {metric_label}"
    print(f"\nGenerating report: {args.output}")
    generate_dashboard_report(result, str(args.output), title=title)
    print(f"Dashboard saved to {args.output}")


def main() -> None:
    """Main entry point for CLI."""
    args = parse_args()

    if not args.command:
        print("Error: No command specified", file=sys.stderr)
        print("Use 'colorcast --help' for usage information", file=sys.stderr)
        sys.exit(1)

    try:
        if args.command == "transfer":
            cmd_transfer(args)
        elif args.command == "batch":
            cmd_batch(args)
        elif args.command == "list-methods":
            cmd_list_methods()
        elif args.command == "info":
            cmd_info(args)
        elif args.command == "dashboard":
            cmd_dashboard(args)
        else:
            print(f"Error: Unknown command '{args.command}'", file=sys.stderr)
            sys.exit(1)
    except (ValueError, FileNotFoundError, ValidationError, ImageProcessingError) as e:
        print(f"Error: {e}", file=sys.stderr)
        if getattr(args, "verbose", False):
            import traceback

            traceback.print_exc(file=sys.stderr)
        sys.exit(2)
    except Exception:  # noqa: BLE001 — last-resort handler; prints a generic message to stderr
        print("Error: An unexpected internal error occurred.", file=sys.stderr)
        if getattr(args, "verbose", False):
            import traceback

            traceback.print_exc(file=sys.stderr)
        sys.exit(3)


def gui_main() -> None:
    """Main entry point for the graphical interface."""
    import argparse

    from colorcast import __version__

    parser = argparse.ArgumentParser(
        prog="colorcast-gui",
        description="Launch the ColorCast graphical interface.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"ColorCast {__version__}",
    )
    parser.parse_args()

    from colorcast.gui import main as _run_gui

    _run_gui()


if __name__ == "__main__":
    gui_main()
