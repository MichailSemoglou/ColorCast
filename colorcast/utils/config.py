"""Configuration management for ColorCast."""

import json
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, asdict, field


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
    cache_size: int = 3
    enable_parallel: bool = True

    # File settings
    last_content_dir: str = ""
    last_style_dir: str = ""
    output_format: str = "png"

    # Custom transfer methods (plugins)
    custom_methods: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        """
        Save configuration to file.

        Args:
            path: Path to save configuration file
        """
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ColorCastConfig":
        """
        Load configuration from file.

        Args:
            path: Path to configuration file

        Returns:
            ColorCastConfig instance
        """
        if not path.exists():
            return cls()

        with open(path, "r") as f:
            data = json.load(f)

        return cls(**data)

    def get_config_path(self) -> Path:
        """
        Get default configuration file path.

        Returns:
            Path to default configuration file
        """
        config_dir = Path.home() / ".colorcast"
        config_dir.mkdir(exist_ok=True)
        return config_dir / "config.json"