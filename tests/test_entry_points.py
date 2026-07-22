"""Smoke tests for CLI and GUI entry points."""

import subprocess
import sys
from pathlib import Path

import pytest


class TestEntryPoints:
    """Smoke tests for installed and module entry points."""

    def test_cli_help(self):
        """`colorcast --help` exits cleanly and prints usage."""
        result = subprocess.run(
            ["colorcast", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "usage: colorcast" in result.stdout

    def test_gui_help(self):
        """`colorcast-gui --help` exits cleanly and prints usage."""
        result = subprocess.run(
            ["colorcast-gui", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "colorcast-gui" in result.stdout
        assert "graphical interface" in result.stdout.lower()

    def test_module_help(self):
        """`python -m colorcast --help` exits cleanly and prints GUI usage."""
        result = subprocess.run(
            [sys.executable, "-m", "colorcast", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "colorcast-gui" in result.stdout
        assert "graphical interface" in result.stdout.lower()

    def test_root_script_help(self):
        """`python colorcast.py --help` exits cleanly via the compatibility shim."""
        root_script = Path(__file__).parent.parent / "colorcast.py"
        result = subprocess.run(
            [sys.executable, str(root_script), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "colorcast-gui" in result.stdout

    def test_main_functions_are_exposed(self):
        """Both CLI and GUI launchers are importable from the package."""
        from colorcast.__main__ import gui_main, main

        assert callable(main)
        assert callable(gui_main)

    def test_entry_points_registered(self):
        """Console scripts are registered in package metadata."""
        from importlib.metadata import entry_points

        scripts = entry_points(group="console_scripts")
        names = {ep.name for ep in scripts}
        assert "colorcast" in names
        assert "colorcast-gui" in names

    def test_cli_transfer_simulator_without_style(self, tmp_path):
        """A simulator method runs without a style image argument."""
        import numpy as np
        from skimage import io

        content_path = tmp_path / "content.png"
        output_path = tmp_path / "out.png"
        io.imsave(
            content_path, (np.random.rand(16, 16, 3) * 255).astype(np.uint8)
        )

        result = subprocess.run(
            [
                "colorcast",
                "transfer",
                str(content_path),
                "-m",
                "simulate_protanopia",
                "-o",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0
        assert output_path.exists()

    def test_cli_transfer_requires_style_for_transfer_methods(self, tmp_path):
        """Omitting the style image fails with a clear error."""
        import numpy as np
        from skimage import io

        content_path = tmp_path / "content.png"
        io.imsave(
            content_path, (np.random.rand(16, 16, 3) * 255).astype(np.uint8)
        )

        result = subprocess.run(
            [
                "colorcast",
                "transfer",
                str(content_path),
                "-m",
                "histogram",
                "-o",
                str(tmp_path / "out.png"),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 1
        assert "requires a style image" in result.stdout

    def test_gui_import_does_not_configure_root_logger(self):
        """Importing colorcast.gui must not add handlers to the root logger."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import logging, colorcast.gui; "
                "raise SystemExit(0 if not logging.getLogger().handlers else 1)",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0
