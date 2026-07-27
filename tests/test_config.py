"""Tests for ColorCastConfig configuration management."""

import json
import pytest
from pathlib import Path
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

    def test_default_cache_size(self):
        cfg = ColorCastConfig()
        assert cfg.cache_size == 3

    def test_default_enable_parallel(self):
        cfg = ColorCastConfig()
        assert cfg.enable_parallel is True

    def test_default_output_format(self):
        cfg = ColorCastConfig()
        assert cfg.output_format == "png"

    def test_default_custom_methods(self):
        cfg = ColorCastConfig()
        assert cfg.custom_methods == {}


class TestColorCastConfigSaveLoad:
    """Tests for save and load roundtrips."""

    def test_save_and_load_roundtrip(self, tmp_path):
        cfg = ColorCastConfig(window_width=800, window_height=600)
        path = tmp_path / "config.json"
        cfg.save(path)

        loaded = ColorCastConfig.load(path)
        assert loaded.window_width == 800
        assert loaded.window_height == 600

    def test_load_missing_file_returns_defaults(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        cfg = ColorCastConfig.load(path)
        assert isinstance(cfg, ColorCastConfig)
        assert cfg.window_width == 1000

    def test_save_produces_valid_json(self, tmp_path):
        cfg = ColorCastConfig()
        path = tmp_path / "config.json"
        cfg.save(path)

        with open(path, "r") as f:
            data = json.load(f)
        assert json.loads(json.dumps(data)) == data

    def test_save_preserves_all_fields(self, tmp_path):
        cfg = ColorCastConfig(
            window_width=1024,
            default_method="meanstd",
            default_intensity=0.5,
            cache_size=10,
        )
        path = tmp_path / "config.json"
        cfg.save(path)

        loaded = ColorCastConfig.load(path)
        assert loaded.window_width == 1024
        assert loaded.default_method == "meanstd"
        assert loaded.default_intensity == pytest.approx(0.5)
        assert loaded.cache_size == 10

    def test_load_filters_unknown_keys(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps({"window_width": 777, "bogus_field": "intruder", "another_junk": 42})
        )

        cfg = ColorCastConfig.load(path)
        assert cfg.window_width == 777
        assert not hasattr(cfg, "bogus_field")


class TestColorCastConfigGetConfigPath:
    """Tests for get_config_path."""

    @pytest.fixture(autouse=True)
    def isolate_home(self, monkeypatch, tmp_path):
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    def test_returns_path_object(self):
        cfg = ColorCastConfig()
        path = cfg.get_config_path()
        assert isinstance(path, Path)

    def test_returns_expected_filename(self):
        cfg = ColorCastConfig()
        path = cfg.get_config_path()
        assert path.name == "config.json"

    def test_returns_expected_directory(self):
        cfg = ColorCastConfig()
        path = cfg.get_config_path()
        assert path.parent.name == ".colorcast"

    def test_directory_is_created(self):
        cfg = ColorCastConfig()
        path = cfg.get_config_path()
        assert path.parent.exists()
        assert path.parent.is_dir()
