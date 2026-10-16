from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class ConfigManager:
    """Centralized configuration for models, camera sources, recording, and performance."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = Path(base_dir or Path(__file__).resolve().parent)
        self.settings_path = self.base_dir / "app_settings.json"
        self.settings: Dict[str, Any] = {
            "app": {
                "name": "Object Detection & Recognition",
                "version": "1.0.0",
                "build": "alpha-1",
                "language": "en",
                "theme": "dark",
                "window": {"width": 1280, "height": 720},
            },
            "model": "yolov8n.pt",
            "confidence_threshold": 0.50,
            "image_size": 640,
            "tracker": "bytetrack.yaml",
            "recording_quality": 28,
            "video_codec": "mp4v",
            "output_paths": {
                "recordings": str(self.base_dir / "recordings"),
                "captures": str(self.base_dir / "captures"),
                "reports": str(self.base_dir / "reports"),
                "logs": str(self.base_dir / "logs"),
                "sessions": str(self.base_dir / "sessions"),
            },
            "dashboard_refresh_rate": 2.0,
            "alert_settings": {},
            "heatmap_settings": {"enabled": True, "opacity": 0.45},
            "camera_resolution": {"width": 640, "height": 480},
            "thread_count": 4,
            "device": "cpu",
            "logging": {"level": "INFO", "rotation_mb": 2},
            "cameras": [],
        }
        self._load_settings()

    def _load_settings(self) -> None:
        try:
            if self.settings_path.exists():
                loaded = json.loads(self.settings_path.read_text())
                self.settings = self._merge_settings(self.settings, loaded)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            self._save_settings()

    def _merge_settings(self, defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(defaults)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._merge_settings(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _save_settings(self) -> None:
        try:
            self.settings_path.write_text(json.dumps(self.settings, indent=2))
        except OSError:
            return

    def save(self) -> None:
        self._save_settings()

    def get(self, key: str, default: Any = None) -> Any:
        value = self.settings
        for part in key.split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        parts = key.split(".")
        target = self.settings
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value
        self._save_settings()

    def add_camera(self, camera: Dict[str, Any]) -> None:
        cameras = self.settings.setdefault("cameras", [])
        cameras.append(camera)
        self._save_settings()

    def remove_camera(self, name: str) -> None:
        cameras = self.settings.get("cameras", [])
        self.settings["cameras"] = [camera for camera in cameras if camera.get("name") != name]
        self._save_settings()

    def get_cameras(self) -> List[Dict[str, Any]]:
        return list(self.settings.get("cameras", []))
