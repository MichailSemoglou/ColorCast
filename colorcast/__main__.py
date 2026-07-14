"""Command-line interface for ColorCast."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from colorcast import (
    load_image,
    save_image,
    match_histograms_multichannel,
    color_transfer_meanstd,
    lut_transfer_with_curve,
    selective_color_transfer,
    blend_images,
    registry,
)
from colorcast.processing.batch import BatchProcessor
from colorcast.processing.image_loader import ALLOWED_IMAGE_EXTENSIONS


def parse_args():
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

    # Transfer command
    transfer_parser = subparsers.add_parser(
        "transfer",
        help="Apply color transfer between two images",
    )
    transfer_parser.add_argument(
        "content",
        type=Path,
        help="Path to content image",
    )
    transfer_parser.add_argument(
        "style",
        type=Path,
        help="Path to style image",
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
        type=float,
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
        "content_dir",
        type=Path,
        help="Directory containing content images",
    )
    batch_parser.add_argument(
        "style",
        type=Path,
        help="Path to style image",
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

    return parser.parse_args()


def get_transfer_function(method_id: str):
    """
    Get transfer function by method ID.

    Args:
        method_id: Method identifier

    Returns:
        Transfer function
    """
    methods = {
        "histogram": match_histograms_multichannel,
        "meanstd": color_transfer_meanstd,
        "lut_linear": lambda s, r: lut_transfer_with_curve(s, r, "linear"),
        "lut_scurve": lambda s, r: lut_transfer_with_curve(s, r, "s-curve"),
        "lut_contrast": lambda s, r: lut_transfer_with_curve(s, r, "contrast"),
        "selective_shadows": lambda s, r: selective_color_transfer(
            s, r, mode="shadows"
        ),
        "selective_midtones": lambda s, r: selective_color_transfer(
            s, r, mode="midtones"
        ),
        "selective_highlights": lambda s, r: selective_color_transfer(
            s, r, mode="highlights"
        ),
    }
    return methods.get(method_id, match_histograms_multichannel)


def cmd_transfer(args):
    """Handle transfer command."""
    print(f"Loading content image: {args.content}")
    content = load_image(str(args.content))
    print(f"Loading style image: {args.style}")
    style = load_image(str(args.style))

    print(f"Applying {args.method} transfer...")
    transfer_func = get_transfer_function(args.method)

    # Apply transfer with parameters
    if args.method.startswith("selective"):
        result = transfer_func(content, style)
    else:
        result = transfer_func(content, style)

    # Apply intensity blending
    if args.intensity < 1.0:
        print(f"Blending with intensity: {args.intensity:.2f}")
        result = blend_images(content, result, args.intensity)

    # Save result
    print(f"Saving result to: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(result, str(args.output))

    print("Transfer complete!")


def cmd_batch(args):
    """Handle batch command."""
    print(f"Loading style image: {args.style}")
    style = load_image(str(args.style))

    print(f"Starting batch processing...")
    print(f"Content directory: {args.content_dir}")
    print(f"Output directory: {args.output}")
    print(f"Method: {args.method}")
    print(f"Workers: {args.workers}")

    transfer_func = get_transfer_function(args.method)

    # Progress callback
    def progress_callback(processed, total):
        percent = (processed / total) * 100
        print(f"Progress: {processed}/{total} ({percent:.1f}%)")

    # Process batch
    processor = BatchProcessor(
        transfer_method=transfer_func,
        max_workers=args.workers,
        progress_callback=progress_callback,
    )

    results = processor.process_directory(
        content_dir=args.content_dir,
        style_image=args.style,
        output_dir=args.output,
        pattern=args.pattern,
    )

    # Report results
    print(f"\nBatch processing complete!")
    print(f"Successfully processed: {len(results)} files")
    if processor.failed_files:
        print(f"Failed: {len(processor.failed_files)} files")
        for failed_path, error in processor.failed_files[:5]:
            print(f"  - {failed_path.name}: {error}")
        if len(processor.failed_files) > 5:
            print(f"  ... and {len(processor.failed_files) - 5} more")


def cmd_list_methods():
    """Handle list-methods command."""
    print("Available transfer methods:")
    print("-" * 50)

    methods = registry.list_methods()
    for method_id, method_name in sorted(methods.items()):
        print(f"  {method_id:25} - {method_name}")

    print("-" * 50)
    print(f"Total: {len(methods)} methods")


def cmd_info(args):
    """Handle info command."""
    from colorcast import __version__, __author__, __license__

    print("ColorCast - Advanced Color Transfer Suite")
    print("-" * 50)

    if args.version:
        print(f"Version: {__version__}")
        print(f"Author: {__author__}")
        print(f"License: {__license__}")
    else:
        print("ColorCast is a sophisticated toolkit for color and")
        print("style transfer between images.")
        print()
        print(f"Version: {__version__}")
        print(f"Author: {__author__}")
        print(f"License: {__license__}")
        print()
        print("Supported image formats:")
        for ext in ALLOWED_IMAGE_EXTENSIONS:
            print(f"  {ext}")
        print()
        print("Use 'colorcast --help' for usage information.")
        print("Visit https://github.com/MichailSemoglou/ColorCast for more info.")


def main():
    """Main entry point for CLI."""
    args = parse_args()

    if not args.command:
        print("Error: No command specified")
        print("Use 'colorcast --help' for usage information")
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
        else:
            print(f"Error: Unknown command '{args.command}'")
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def gui_main():
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