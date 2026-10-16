from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class DashboardManager:
    """Lightweight dashboard and historical analytics manager for the desktop app."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = Path(base_dir or Path(__file__).resolve().parent)
        self.db_path = self.base_dir / "dashboard.db"
        self.reports_dir = self.base_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.settings_path = self.base_dir / "dashboard_settings.json"
        self.dashboard_settings = {
            "refresh_rate_hz": 2.0,
            "heatmap_resolution": 24,
            "graph_history_length": 60,
            "report_location": str(self.reports_dir),
            "auto_report_generation": False,
            "analytics_sampling_rate": 5,
        }
        self._load_settings()
        self._initialize_database()
        self.last_refresh = 0.0
        self.heatmap_enabled = True
        self.heatmap_opacity = 0.45
        self.heatmap_points: List[Tuple[int, int]] = []
        self.heatmap_weights: List[float] = []
        self.analytics_cache: Dict[str, Any] = {}

    def _initialize_database(self) -> None:
        try:
            with self._connect_db() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS DASHBOARD_EVENTS (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        details TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS HEATMAP_DATA (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        points TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS REPORTS (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        report_type TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        range_start TEXT,
                        range_end TEXT,
                        path TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS DASHBOARD_PREFERENCES (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
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
                self.dashboard_settings.update(json.loads(self.settings_path.read_text()))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            self._save_settings()

    def _save_settings(self) -> None:
        try:
            self.settings_path.write_text(json.dumps(self.dashboard_settings, indent=2))
        except OSError:
            return

    def update_dashboard(self, summary: Dict[str, Any], detections: List[Dict[str, Any]], registered_objects: List[Dict[str, Any]], alerts: List[Dict[str, Any]], frame_shape: Tuple[int, int, int]) -> Dict[str, Any]:
        now = time.time()
        if now - self.last_refresh < 1.0 / max(1.0, self.dashboard_settings.get("refresh_rate_hz", 2.0)):
            return self.analytics_cache
        self.last_refresh = now

        object_counts: Dict[str, int] = {}
        for detection in detections:
            class_name = detection.get("class_name", "object")
            object_counts[class_name] = object_counts.get(class_name, 0) + 1

        self.analytics_cache = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "fps": summary.get("fps", 0),
            "tracked_objects": summary.get("tracked_objects", 0),
            "registered_objects": len(registered_objects),
            "unknown_objects": max(0, summary.get("tracked_objects", 0) - len(registered_objects)),
            "active_alerts": len(alerts),
            "active_zones": summary.get("zone_count", 0),
            "active_lines": summary.get("line_count", 0),
            "object_counts": object_counts,
            "recognition_success_rate": 0.0,
            "unknown_percentage": 0.0,
            "average_speed": summary.get("average_scene_speed", 0.0),
            "fastest_object": summary.get("fastest_object"),
            "closest_object": summary.get("closest_object_pair"),
            "zone_occupancy": summary.get("zone_counts", {}),
            "line_crossings": summary.get("line_counts", {}),
            "collision_warnings": summary.get("predicted_collisions", 0),
            "safe_distance_violations": summary.get("safe_distance_violations", 0),
            "intrusion_events": summary.get("line_counts", {}).get("TOTAL", 0),
            "heatmap_points": list(self.heatmap_points),
            "frame_shape": frame_shape,
        }
        self._record_event("dashboard_update", json.dumps(self.analytics_cache))
        return self.analytics_cache

    def toggle_heatmap(self) -> None:
        self.heatmap_enabled = not self.heatmap_enabled

    def reset_heatmap(self) -> None:
        self.heatmap_points = []
        self.heatmap_weights = []

    def add_heatmap_point(self, point: Tuple[int, int], weight: float = 1.0) -> None:
        self.heatmap_points.append(point)
        self.heatmap_weights.append(weight)
        if len(self.heatmap_points) > 400:
            self.heatmap_points = self.heatmap_points[-400:]
            self.heatmap_weights = self.heatmap_weights[-400:]

    def draw_heatmap(self, frame: np.ndarray) -> None:
        if not self.heatmap_enabled or not self.heatmap_points:
            return
        overlay = frame.copy()
        for point in self.heatmap_points[-100:]:
            cv2.circle(overlay, point, 4, (0, 255, 255), -1)
        cv2.addWeighted(overlay, self.heatmap_opacity, frame, 1.0 - self.heatmap_opacity, 0, frame)

    def save_heatmap_image(self, frame: np.ndarray, name: str = "heatmap") -> Optional[str]:
        try:
            path = self.reports_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(str(path), frame)
            return str(path)
        except Exception:
            return None

    def export_report(self, report_type: str, range_start: Optional[str] = None, range_end: Optional[str] = None) -> Optional[str]:
        try:
            report_path = self.reports_dir / f"{report_type.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            report_path.write_text(f"{report_type} report\nGenerated: {datetime.now()}\n")
            with self._connect_db() as conn:
                conn.execute(
                    "INSERT INTO REPORTS (report_type, created_at, range_start, range_end, path) VALUES (?, ?, ?, ?, ?)",
                    (report_type, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), range_start, range_end, str(report_path)),
                )
            return str(report_path)
        except OSError:
            return None

    def _record_event(self, event_type: str, details: str) -> None:
        try:
            with self._connect_db() as conn:
                conn.execute(
                    "INSERT INTO DASHBOARD_EVENTS (timestamp, event_type, details) VALUES (?, ?, ?)",
                    (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), event_type, details),
                )
        except sqlite3.Error:
            return

    def get_recent_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        try:
            with self._connect_db() as conn:
                rows = conn.execute(
                    "SELECT timestamp, event_type, details FROM DASHBOARD_EVENTS ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                return [{"timestamp": row[0], "event_type": row[1], "details": row[2]} for row in rows]
        except sqlite3.Error:
            return []
