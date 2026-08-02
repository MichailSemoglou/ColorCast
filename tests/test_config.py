"""Tests for ColorCastConfig configuration management."""

import json
from pathlib import Path

import pytest

from colorcast.utils.config import ColorCastConfig


class TestColorCastConfigDefaults:
    """Tests for default configuration values."""

    def test_default_window_width(self):
        cfg = ColorCastConfig()
        assert cfg.window_width == 1000

    def test_default_window_height(self):
        cfg = ColorCastConfig()
        assert cfg.window_height == 700

    def test_default_preview_size(self):
        cfg = ColorCastConfig()
        assert cfg.preview_size == 300

    def test_default_method(self):
        cfg = ColorCastConfig()
        assert cfg.default_method == "histogram"

    def test_default_intensity(self):
        cfg = ColorCastConfig()
        assert cfg.default_intensity == pytest.approx(0.85)

    def test_default_slider_debounce(self):
        cfg = ColorCastConfig()
        assert cfg.slider_debounce_ms == 50

    def test_default_max_image_dimension(self):
        cfg = ColorCastConfig()
        assert cfg.max_image_dimension == 4096

    def test_default_enable_parallel(self):
        cfg = ColorCastConfig()
        assert cfg.enable_parallel is True

    def test_default_output_format(self):
        cfg = ColorCastConfig()
        assert cfg.output_format == "png"

    def test_default_custom_methods(self):
        cfg = ColorCastConfig()
        assert cfg.custom_methods == {}


class TestColorCastConfigPersistence:
    """Tests for save/load round-trip and validation."""

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        cfg = ColorCastConfig(window_width=1920, default_method="meanstd")
        path = cfg.save(tmp_path / "config.json")
        loaded = ColorCastConfig.load(path)
        assert loaded.window_width == 1920
        assert loaded.default_method == "meanstd"
        assert loaded.window_height == 700  # unmodified default

    def test_save_creates_parent_directories(self, tmp_path: Path):
        cfg = ColorCastConfig()
        path = tmp_path / "deep" / "nested" / "config.json"
        cfg.save(path)
        assert path.exists()

    def test_load_ignores_unknown_keys(self, tmp_path: Path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps({"window_width": 800, "nonexistent_field": "should_be_ignored"}),
            encoding="utf-8",
        )
        cfg = ColorCastConfig.load(path)
        assert cfg.window_width == 800

    def test_load_raises_on_bad_type(self, tmp_path: Path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps({"window_width": "abc"}),
            encoding="utf-8",
        )
        with pytest.raises(TypeError, match="window_width"):
            ColorCastConfig.load(path)

    def test_load_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            ColorCastConfig.load(tmp_path / "nonexistent.json")

    def test_load_partial_overlay(self, tmp_path: Path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps({"window_width": 1024, "output_format": "jpg"}),
            encoding="utf-8",
        )
        cfg = ColorCastConfig.load(path)
        assert cfg.window_width == 1024
        assert cfg.output_format == "jpg"
        assert cfg.window_height == 700  # default

    def test_get_config_path_returns_path(self):
        path = ColorCastConfig.get_config_path()
        assert isinstance(path, Path)
        assert path.name == "config.json"

    def test_save_returns_written_path(self, tmp_path: Path):
        cfg = ColorCastConfig()
        returned = cfg.save(tmp_path / "cfg.json")
        assert returned == tmp_path / "cfg.json"
