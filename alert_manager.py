from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class AlertManager:
    """Centralized, modular alert generation and management for scene analytics."""

    def __init__(self, base_dir: Optional[Path] = None, camera_source: str = "camera_0") -> None:
        self.base_dir = Path(base_dir or Path(__file__).resolve().parent)
        self.camera_source = camera_source
        self.settings_path = self.base_dir / "alert_settings.json"
        self.db_path = self.base_dir / "alerts.db"
        self.captures_dir = self.base_dir / "captures"
        self.alerts: List[Dict[str, Any]] = []
        self.alert_counters: Dict[str, int] = {"INFO": 0, "WARNING": 0, "CRITICAL": 0}
        self._alert_state: Dict[str, Dict[str, Any]] = {}
        self._sounds_enabled = True
        self._sound_map = {"INFO": "beep", "WARNING": "beep", "CRITICAL": "beep"}
        self._load_settings()
        self._initialize_database()

    def _initialize_database(self) -> None:
        try:
            with self._connect_db() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ALERTS (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alert_id TEXT NOT NULL UNIQUE,
                        alert_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        camera_source TEXT NOT NULL,
                        track_id INTEGER,
                        object_name TEXT,
                        object_class TEXT,
                        zone TEXT,
                        screenshot TEXT,
                        description TEXT NOT NULL
                    )
                    """
                )
        except sqlite3.Error:
            return

    def _connect_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _load_settings(self) -> None:
        try:
            if self.settings_path.exists():
                data = json.loads(self.settings_path.read_text())
                self._sounds_enabled = bool(data.get("sounds_enabled", True))
                self._sound_map = {
                    "INFO": data.get("sound_info", "beep"),
                    "WARNING": data.get("sound_warning", "beep"),
                    "CRITICAL": data.get("sound_critical", "beep"),
                }
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            self._save_settings()

    def _save_settings(self) -> None:
        try:
            self.settings_path.write_text(
                json.dumps(
                    {
                        "sounds_enabled": self._sounds_enabled,
                        "sound_info": self._sound_map.get("INFO", "beep"),
                        "sound_warning": self._sound_map.get("WARNING", "beep"),
                        "sound_critical": self._sound_map.get("CRITICAL", "beep"),
                    },
                    indent=2,
                )
            )
        except OSError:
            return

    def set_sounds_enabled(self, enabled: bool) -> None:
        self._sounds_enabled = enabled
        self._save_settings()

    def create_alert(
        self,
        alert_type: str,
        severity: str,
        description: str,
        track_id: Optional[int] = None,
        object_name: Optional[str] = None,
        object_class: Optional[str] = None,
        zone: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        frame: Optional[np.ndarray] = None,
        annotated_frame: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        alert_id = f"{alert_type}:{track_id or 0}:{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        if self._is_duplicate(alert_type, track_id, zone, description):
            return None

        alert = {
            "alert_id": alert_id,
            "alert_type": alert_type,
            "severity": severity,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "camera_source": self.camera_source,
            "track_id": track_id,
            "object_name": object_name or "Unknown",
            "object_class": object_class or "Unknown",
            "zone": zone,
            "screenshot": None,
            "description": description,
            "metadata": metadata or {},
        }
        self.alerts.insert(0, alert)
        self.alerts = self.alerts[:100]
        self.alert_counters[severity] = self.alert_counters.get(severity, 0) + 1
        self._alert_state[alert_id] = alert
        self._persist_alert(alert, frame, annotated_frame)
        self._play_sound(severity)
        return alert

    def _is_duplicate(self, alert_type: str, track_id: Optional[int], zone: Optional[str], description: str) -> bool:
        for alert in self.alerts:
            if alert.get("alert_type") != alert_type:
                continue
            if alert.get("track_id") != track_id:
                continue
            if alert.get("zone") != zone:
                continue
            if alert.get("description") == description:
                return True
        return False

    def _persist_alert(self, alert: Dict[str, Any], frame: Optional[np.ndarray], annotated_frame: Optional[np.ndarray]) -> None:
        screenshot_path = None
        if alert.get("severity") == "CRITICAL":
            screenshot_path = self._capture_screenshot(frame, annotated_frame, alert["alert_type"])
        try:
            with self._connect_db() as conn:
                conn.execute(
                    """
                    INSERT INTO ALERTS (
                        alert_id, alert_type, severity, timestamp, camera_source, track_id, object_name,
                        object_class, zone, screenshot, description
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        alert["alert_id"],
                        alert["alert_type"],
                        alert["severity"],
                        alert["timestamp"],
                        alert["camera_source"],
                        alert.get("track_id"),
                        alert.get("object_name"),
                        alert.get("object_class"),
                        alert.get("zone"),
                        screenshot_path,
                        alert["description"],
                    ),
                )
        except sqlite3.Error:
            return
        alert["screenshot"] = screenshot_path

    def _capture_screenshot(self, frame: Optional[np.ndarray], annotated_frame: Optional[np.ndarray], alert_type: str) -> Optional[str]:
        if frame is None:
            return None
        try:
            capture_dir = self.captures_dir / datetime.now().strftime("%Y-%m-%d")
            capture_dir.mkdir(parents=True, exist_ok=True)
            file_name = f"{datetime.now().strftime('%H-%M-%S')}_{alert_type.lower().replace(' ', '_')}.jpg"
            path = capture_dir / file_name
            output_frame = annotated_frame if annotated_frame is not None else frame
            cv2.imwrite(str(path), output_frame)
            return str(path)
        except Exception:
            return None

    def _play_sound(self, severity: str) -> None:
        if not self._sounds_enabled:
            return
        sound = self._sound_map.get(severity, "beep")
        if sound in {"", None}:
            return
        try:
            if sys.platform == "darwin":
                subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                os.system("printf '\a'")
        except Exception:
            return

    def get_alerts(self) -> List[Dict[str, Any]]:
        return list(self.alerts)

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total": len(self.alerts),
            "info": self.alert_counters.get("INFO", 0),
            "warning": self.alert_counters.get("WARNING", 0),
            "critical": self.alert_counters.get("CRITICAL", 0),
            "latest": self.alerts[0] if self.alerts else None,
        }
