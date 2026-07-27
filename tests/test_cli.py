"""Tests for the command-line interface functions.

These tests exercise the CLI functions directly (parse_args, cmd_transfer,
cmd_batch, cmd_list_methods, cmd_info, main) rather than through subprocess.
"""

import argparse
import sys
import numpy as np
import pytest
from unittest import mock

from colorcast.__main__ import (
    parse_args,
    cmd_transfer,
    cmd_batch,
    cmd_list_methods,
    cmd_info,
    main,
)
from colorcast.processing.image_loader import ALLOWED_IMAGE_EXTENSIONS


def _make_args(**overrides):
    """Build an argparse.Namespace with sensisible defaults."""
    import pathlib
    defaults = {
        "command": "transfer",
        "content": pathlib.Path("fake_content.jpg"),
        "style": pathlib.Path("fake_style.jpg"),
        "output": pathlib.Path("fake_output.jpg"),
        "method": "histogram",
        "intensity": 1.0,
        "shadow_threshold": 0.3,
        "highlight_threshold": 0.7,
        "content_dir": pathlib.Path("fake_content_dir"),
        "workers": 4,
        "pattern": "*.jpg",
        "version": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_transfer_command(self):
        argv = ["colorcast", "transfer", "content.jpg", "style.jpg", "-o", "output.jpg"]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()
            assert args.command == "transfer"
            assert args.method == "histogram"

    def test_transfer_command_with_method(self):
        argv = ["colorcast", "transfer", "content.jpg", "style.jpg", "-o", "output.jpg", "-m", "meanstd"]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()
            assert args.command == "transfer"
            assert args.method == "meanstd"

    def test_batch_command(self):
        argv = ["colorcast", "batch", "content_dir", "style.jpg", "-o", "output_dir"]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()
            assert args.command == "batch"

    def test_list_methods_command(self):
        argv = ["colorcast", "list-methods"]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()
            assert args.command == "list-methods"

    def test_info_command(self):
        argv = ["colorcast", "info"]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()
            assert args.command == "info"

    def test_info_command_version(self):
        argv = ["colorcast", "info", "--version"]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()
            assert args.command == "info"
            assert args.version is True


class TestCmdTransfer:
    """Tests for the transfer command handler."""

    def test_transfer_with_simulator_requires_no_style(self, tmp_path):
        source = np.random.rand(50, 50, 3).astype(np.float32)
        content_path = tmp_path / "content.jpg"
        output_path = tmp_path / "output.png"

        import imageio.v3 as iio
        iio.imwrite(str(content_path), (source * 255).astype(np.uint8))

        args = _make_args(
            command="transfer",
            content=content_path,
            style=None,
            output=output_path,
            method="simulate_protanopia",
            intensity=1.0,
        )
        cmd_transfer(args)
        assert output_path.exists()

    def test_transfer_with_style(self, tmp_path):
        source = np.random.rand(50, 50, 3).astype(np.float32)
        reference = np.random.rand(50, 50, 3).astype(np.float32)
        content_path = tmp_path / "content.jpg"
        style_path = tmp_path / "style.jpg"
        output_path = tmp_path / "output.png"

        import imageio.v3 as iio
        iio.imwrite(str(content_path), (source * 255).astype(np.uint8))
        iio.imwrite(str(style_path), (reference * 255).astype(np.uint8))

        args = _make_args(
            command="transfer",
            content=content_path,
            style=style_path,
            output=output_path,
            method="histogram",
            intensity=1.0,
        )
        cmd_transfer(args)
        assert output_path.exists()

    def test_transfer_with_intensity_blend(self, tmp_path):
        source = np.random.rand(50, 50, 3).astype(np.float32)
        reference = np.random.rand(50, 50, 3).astype(np.float32)
        content_path = tmp_path / "content.jpg"
        style_path = tmp_path / "style.jpg"
        output_path = tmp_path / "output.png"

        import imageio.v3 as iio
        iio.imwrite(str(content_path), (source * 255).astype(np.uint8))
        iio.imwrite(str(style_path), (reference * 255).astype(np.uint8))

        args = _make_args(
            command="transfer",
            content=content_path,
            style=style_path,
            output=output_path,
            method="histogram",
            intensity=0.5,
        )
        cmd_transfer(args)
        assert output_path.exists()

    def test_transfer_missing_style_for_style_method_raises(self, tmp_path):
        source = np.random.rand(50, 50, 3).astype(np.float32)
        content_path = tmp_path / "content.jpg"
        output_path = tmp_path / "output.png"

        import imageio.v3 as iio
        iio.imwrite(str(content_path), (source * 255).astype(np.uint8))

        args = _make_args(
            command="transfer",
            content=content_path,
            style=None,
            output=output_path,
            method="histogram",
        )
        with pytest.raises(ValueError, match="requires a style image"):
            cmd_transfer(args)

    def test_transfer_creates_output_parent_dirs(self, tmp_path):
        source = np.random.rand(50, 50, 3).astype(np.float32)
        reference = np.random.rand(50, 50, 3).astype(np.float32)
        content_path = tmp_path / "content.jpg"
        style_path = tmp_path / "style.jpg"
        output_path = tmp_path / "subdir" / "nested" / "output.png"

        import imageio.v3 as iio
        iio.imwrite(str(content_path), (source * 255).astype(np.uint8))
        iio.imwrite(str(style_path), (reference * 255).astype(np.uint8))

        args = _make_args(
            command="transfer",
            content=content_path,
            style=style_path,
            output=output_path,
            method="histogram",
            intensity=1.0,
        )
        cmd_transfer(args)
        assert output_path.exists()


class TestCmdListMethods:
    """Tests for the list-methods command."""

    def test_list_methods_runs(self, capsys):
        cmd_list_methods()
        captured = capsys.readouterr()
        assert "Available transfer methods" in captured.out
        assert "histogram" in captured.out


class TestCmdInfo:
    """Tests for the info command."""

    def test_info_version(self, capsys):
        args = argparse.Namespace(version=True)
        cmd_info(args)
        captured = capsys.readouterr()
        assert "Version:" in captured.out

    def test_info_full(self, capsys):
        args = argparse.Namespace(version=False)
        cmd_info(args)
        captured = capsys.readouterr()
        assert "ColorCast" in captured.out
        for ext in ALLOWED_IMAGE_EXTENSIONS:
            assert ext in captured.out


class TestMain:
    """Tests for the CLI main entry point."""

    def test_main_no_command_exits(self):
        argv = ["colorcast"]
        with mock.patch.object(sys, "argv", argv):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_main_info_command(self, capsys):
        argv = ["colorcast", "info", "--version"]
        with mock.patch.object(sys, "argv", argv):
            main()
        captured = capsys.readouterr()
        assert "Version:" in captured.out

    def test_main_list_methods_command(self, capsys):
        argv = ["colorcast", "list-methods"]
        with mock.patch.object(sys, "argv", argv):
            main()
        captured = capsys.readouterr()
        assert "Available transfer methods" in captured.out

    def test_main_unknown_command_exits(self):
        argv = ["colorcast", "bogus-command"]
        with mock.patch.object(sys, "argv", argv):
            with pytest.raises(SystemExit):
                main()

    def test_main_transfer_simulate(self, tmp_path, capsys):
        source = np.random.rand(50, 50, 3).astype(np.float32)
        content_path = tmp_path / "content.jpg"
        output_path = tmp_path / "output.png"

        import imageio.v3 as iio
        iio.imwrite(str(content_path), (source * 255).astype(np.uint8))

        argv = [
            "colorcast", "transfer",
            str(content_path),
            "-o", str(output_path),
            "-m", "simulate_protanopia",
        ]
        with mock.patch.object(sys, "argv", argv):
            main()
        assert output_path.exists()
