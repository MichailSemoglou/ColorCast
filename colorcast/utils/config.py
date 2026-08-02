"""Configuration management for ColorCast."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class ColorCastConfig:
    """Application configuration."""

    # UI settings
    window_width: int = 1000
    window_height: int = 700
    preview_size: int = 300

    # Processing settings
    default_method: str = "histogram"
    default_intensity: float = 0.85
    slider_debounce_ms: int = 50

    # Performance settings
    max_image_dimension: int = 4096
    enable_parallel: bool = True  # honoured by BatchProcessor.enable_parallel

    # File settings
    last_content_dir: str = ""
    last_style_dir: str = ""
    output_format: str = "png"

    # Custom transfer methods (plugins)
    custom_methods: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def get_config_path() -> Path:
        """
        Return the default configuration file path.

        Platform convention for user config directories:
        - macOS: ``~/Library/Application Support/ColorCast/config.json``
        - Linux: ``~/.config/ColorCast/config.json``
        - Windows: ``%APPDATA%/ColorCast/config.json``
        """
        import sys

        if sys.platform == "darwin":
            base = Path.home() / "Library" / "Application Support"
        elif sys.platform == "win32":
            base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        else:
            base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        return base / "ColorCast" / "config.json"

    def save(self, path: str | Path | None = None) -> Path:
        """
        Serialize the current configuration to a JSON file.

        Args:
            path: File path.  When None, ``get_config_path()`` is used.

        Returns:
            The resolved path that was written.

        Raises:
            OSError: If the parent directory cannot be created.
        """
        resolved = Path(path) if path is not None else self.get_config_path()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        resolved.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return resolved

    @classmethod
    def load(cls, path: str | Path | None = None) -> ColorCastConfig:
        """
        Load configuration from a JSON file, falling back to defaults.

        Unknown keys in the file are silently ignored.  Values for known
        fields are validated against their declared types; a
        :class:`TypeError` is raised on mismatch (e.g. ``window_width:
        "abc"`` where ``window_width`` is an ``int`` field).

        Args:
            path: File path.  When None, ``get_config_path()`` is used.

        Returns:
            A ``ColorCastConfig`` populated from the file, with defaults
            for any field not present.

        Raises:
            TypeError: If a known field has an incompatible value type.
            FileNotFoundError: If the file does not exist (surfaced
                directly for callers to handle).
        """
        from typing import Any, get_origin, get_type_hints

        resolved = Path(path) if path is not None else cls.get_config_path()
        raw = json.loads(resolved.read_text(encoding="utf-8"))
        known_fields = {f.name: f for f in fields(cls)}
        field_types = get_type_hints(cls)
        filtered: dict[str, Any] = {}
        for key, value in raw.items():
            if key not in known_fields:
                continue
            expected = field_types.get(key, Any)
            if expected is Any:
                filtered[key] = value
                continue
            origin = get_origin(expected) or expected
            if origin is int and isinstance(value, bool):
                raise TypeError(
                    f"{key}: expected {origin.__name__}, " f"got {type(value).__name__} ({value!r})"
                )
            if origin is float and isinstance(value, int) and not isinstance(value, bool):
                filtered[key] = float(value)
                continue
            if not isinstance(value, origin):
                raise TypeError(
                    f"{key}: expected {origin.__name__}, " f"got {type(value).__name__} ({value!r})"
                )
            filtered[key] = value
        return cls(**filtered)


__all__ = ["ColorCastConfig"]
