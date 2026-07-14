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
