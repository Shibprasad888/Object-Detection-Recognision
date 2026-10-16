from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class AnalyticsLine:
    start: Tuple[int, int]
    end: Tuple[int, int]
    name: str = "Line"


@dataclass
class AnalyticsZone:
    points: List[Tuple[int, int]]
    name: str = "Zone"


@dataclass
class AnalyticsTrack:
    track_id: int
    class_name: str
    object_name: str = "Unknown"
    bounding_box: List[int] = field(default_factory=list)
    center: Tuple[int, int] = (0, 0)
    velocity: Tuple[float, float] = (0.0, 0.0)
    direction: str = ""
    trail: List[Tuple[int, int]] = field(default_factory=list)
    current_zone: Optional[str] = None
    previous_zone: Optional[str] = None
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    dwell_time: float = 0.0
    recognition_status: str = "Unknown"
    last_seen: Optional[datetime] = None
    recognition_count: int = 0
    previous_position: Tuple[int, int] = (0, 0)
    movement_distance: float = 0.0
    frame_history: List[Tuple[int, int]] = field(default_factory=list)
    speed_history: List[Dict[str, Any]] = field(default_factory=list)
    current_speed_mps: float = 0.0
    average_speed_mps: float = 0.0
    max_speed_mps: float = 0.0
    acceleration_mps2: float = 0.0
    estimated_distance_m: float = 0.0
    predicted_collision_seconds: Optional[float] = None
    predicted_collision_target: Optional[int] = None
    safe_distance_violation: bool = False


class SceneAnalytics:
    """Lightweight analytics for object movement, counting, zones, and trails."""

    def __init__(self, settings_path: Optional[Path] = None) -> None:
        self.settings_path = settings_path or Path(__file__).resolve().parent / "motion_settings.json"
        self.lines: List[AnalyticsLine] = []
        self.zones: List[AnalyticsZone] = []
        self.tracks: Dict[int, AnalyticsTrack] = {}
        self.line_counts: Dict[str, int] = {"IN": 0, "OUT": 0, "TOTAL": 0}
        self.zone_counts: Dict[str, int] = defaultdict(int)
        self.events: List[Dict[str, Any]] = []
        self.trails_enabled = True
        self.paused = False
        self.last_frame_time: Optional[datetime] = None
        self.pixel_to_meter_ratio = 0.01
        self.reference_object_size_m = 0.20
        self.reference_object_height_px = 100
        self.minimum_safe_distance = 1.5
        self.maximum_prediction_time = 3.0
        self.speed_smoothing_window = 5
        self.distance_threshold = 2.5
        self.trail_length = 30
        self.safe_distance_violations = 0
        self.predicted_collisions = 0
        self._load_settings()

    def create_line(self, start: Tuple[int, int], end: Tuple[int, int], name: str = "Line") -> None:
        if len(self.lines) >= 5:
            return
        self.lines.append(AnalyticsLine(start=start, end=end, name=name))

    def create_zone(self, points: List[Tuple[int, int]], name: str = "Zone") -> None:
        if len(self.zones) >= 8:
            return
        if len(points) < 3:
            return
        self.zones.append(AnalyticsZone(points=points, name=name))

    def reset_counters(self) -> None:
        self.line_counts = {"IN": 0, "OUT": 0, "TOTAL": 0}
        self.zone_counts = defaultdict(int)
        self.events = []

    def set_calibration(
        self,
        pixel_to_meter_ratio: Optional[float] = None,
        reference_object_size_m: Optional[float] = None,
        reference_object_height_px: Optional[float] = None,
        minimum_safe_distance: Optional[float] = None,
        maximum_prediction_time: Optional[float] = None,
        speed_smoothing_window: Optional[int] = None,
        distance_threshold: Optional[float] = None,
        trail_length: Optional[int] = None,
    ) -> None:
        if pixel_to_meter_ratio is not None:
            self.pixel_to_meter_ratio = max(float(pixel_to_meter_ratio), 1e-6)
        if reference_object_size_m is not None:
            self.reference_object_size_m = max(float(reference_object_size_m), 1e-6)
        if reference_object_height_px is not None:
            self.reference_object_height_px = max(float(reference_object_height_px), 1.0)
        if minimum_safe_distance is not None:
            self.minimum_safe_distance = max(float(minimum_safe_distance), 0.0)
        if maximum_prediction_time is not None:
            self.maximum_prediction_time = max(float(maximum_prediction_time), 0.1)
        if speed_smoothing_window is not None:
            self.speed_smoothing_window = max(int(speed_smoothing_window), 1)
        if distance_threshold is not None:
            self.distance_threshold = max(float(distance_threshold), 0.0)
        if trail_length is not None:
            self.trail_length = max(int(trail_length), 5)
        self._save_settings()

    def _load_settings(self) -> None:
        try:
            if self.settings_path.exists():
                data = json.loads(self.settings_path.read_text())
                self.pixel_to_meter_ratio = float(data.get("pixel_to_meter_ratio", self.pixel_to_meter_ratio))
                self.reference_object_size_m = float(data.get("reference_object_size_m", self.reference_object_size_m))
                self.reference_object_height_px = float(data.get("reference_object_height_px", self.reference_object_height_px))
                self.minimum_safe_distance = float(data.get("minimum_safe_distance", self.minimum_safe_distance))
                self.maximum_prediction_time = float(data.get("maximum_prediction_time", self.maximum_prediction_time))
                self.speed_smoothing_window = int(data.get("speed_smoothing_window", self.speed_smoothing_window))
                self.distance_threshold = float(data.get("distance_threshold", self.distance_threshold))
                self.trail_length = int(data.get("trail_length", self.trail_length))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_settings()

    def _save_settings(self) -> None:
        try:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "pixel_to_meter_ratio": self.pixel_to_meter_ratio,
                "reference_object_size_m": self.reference_object_size_m,
                "reference_object_height_px": self.reference_object_height_px,
                "minimum_safe_distance": self.minimum_safe_distance,
                "maximum_prediction_time": self.maximum_prediction_time,
                "speed_smoothing_window": self.speed_smoothing_window,
                "distance_threshold": self.distance_threshold,
                "trail_length": self.trail_length,
            }
            self.settings_path.write_text(json.dumps(data, indent=2))
        except OSError:
            return

    def toggle_trails(self) -> None:
        self.trails_enabled = not self.trails_enabled

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def update_tracks(self, detections: List[Dict[str, Any]], frame_shape: Tuple[int, int, int]) -> None:
        if self.paused:
            return

        current_time = datetime.now()
        self.last_frame_time = current_time
        current_ids = {detection.get("track_id") for detection in detections if detection.get("track_id") is not None}

        for track_id in list(self.tracks.keys()):
            if track_id not in current_ids:
                self._log_event(track_id, "Lost Tracking", "tracking")
                self.tracks.pop(track_id, None)

        for detection in detections:
            track_id = detection.get("track_id")
            if track_id is None:
                continue

            center = tuple(detection.get("center_point", [0, 0]))
            track = self.tracks.get(track_id)
            if track is None:
                track = AnalyticsTrack(track_id=track_id, class_name=detection.get("class_name", "object"))
                self.tracks[track_id] = track
                track.object_name = detection.get("object_name", "Unknown")
                track.recognition_status = detection.get("recognition_status", "Unknown")
                track.last_seen = current_time
                track.recognition_count = 1 if detection.get("recognition_status") == "Known" else 0
                self._log_event(track_id, "Entered Scene", "tracking")

            track.bounding_box = detection.get("bounding_box", [])
            track.center = center
            previous_position = track.previous_position if track.previous_position != (0, 0) else center
            displacement = distance_between(previous_position, center)
            track.movement_distance = displacement
            track.velocity = (center[0] - previous_position[0], center[1] - previous_position[1])
            track.direction = direction_from_delta(track.velocity[0], track.velocity[1])
            track.frame_history = (track.frame_history + [center])[-30:]
            track.last_seen = current_time
            track.previous_position = center
            track.recognition_status = detection.get("recognition_status", "Unknown")
            track.object_name = detection.get("object_name", "Unknown")
            if detection.get("recognition_status") == "Known":
                track.recognition_count += 1

            self._update_motion_metrics(track, displacement, current_time)

            if self.trails_enabled:
                track.trail = (track.trail + [center])[-self.trail_length:]
            else:
                track.trail = []

            self._update_zone_state(track, current_time)
            self._update_line_crossings(track, current_time)
            self._update_safe_distance(track)
            self._update_collision_predictions(track)

    def _update_motion_metrics(self, track: AnalyticsTrack, displacement: float, current_time: datetime) -> None:
        if track.previous_position == (0, 0):
            track.current_speed_mps = 0.0
            track.average_speed_mps = 0.0
            track.max_speed_mps = 0.0
            track.acceleration_mps2 = 0.0
            track.estimated_distance_m = 0.0
            return

        previous_speed = track.current_speed_mps
        pixel_speed = displacement / max(1, self.speed_smoothing_window)
        speed_mps = pixel_speed * self.pixel_to_meter_ratio
        track.current_speed_mps = speed_mps
        track.speed_history.append(
            {
                "timestamp": current_time.strftime("%H:%M:%S"),
                "speed_mps": speed_mps,
                "distance_m": speed_mps * 0.1,
            }
        )
        if len(track.speed_history) > self.speed_smoothing_window:
            track.speed_history = track.speed_history[-self.speed_smoothing_window:]
        average_speed = sum(item["speed_mps"] for item in track.speed_history) / max(1, len(track.speed_history))
        track.average_speed_mps = average_speed
        track.max_speed_mps = max(track.max_speed_mps, speed_mps)
        track.acceleration_mps2 = speed_mps - previous_speed
        track.estimated_distance_m = max(0.0, self.reference_object_size_m * (self.reference_object_height_px / max(1.0, self._estimate_pixel_height(track))))
        self._log_event(track.track_id, f"Speed Updated:{speed_mps:.2f}", "motion")

    def _estimate_pixel_height(self, track: AnalyticsTrack) -> float:
        if not track.bounding_box:
            return float(self.reference_object_height_px)
        x1, y1, x2, y2 = track.bounding_box
        return max(1.0, abs(y2 - y1))

    def _update_safe_distance(self, track: AnalyticsTrack) -> None:
        if not track.center:
            return
        for other in self.tracks.values():
            if other.track_id == track.track_id:
                continue
            distance = distance_between(track.center, other.center)
            if distance <= self.distance_threshold:
                track.safe_distance_violation = True
                other.safe_distance_violation = True
                self.safe_distance_violations += 1
                self.predicted_collisions += 1
                self._log_event(track.track_id, f"Safe Distance Warning:{other.track_id}", "motion")
                return
        track.safe_distance_violation = False

    def _update_collision_predictions(self, track: AnalyticsTrack) -> None:
        if not track.center:
            return
        for other in self.tracks.values():
            if other.track_id == track.track_id or not other.center:
                continue
            dx = other.center[0] - track.center[0]
            dy = other.center[1] - track.center[1]
            dist = math.hypot(dx, dy)
            if dist > 200:
                continue
            relative_velocity = math.hypot(track.velocity[0] - other.velocity[0], track.velocity[1] - other.velocity[1])
            if relative_velocity <= 0:
                continue
            time_to_collision = dist / max(relative_velocity, 1e-6)
            if time_to_collision <= self.maximum_prediction_time:
                track.predicted_collision_seconds = time_to_collision
                track.predicted_collision_target = other.track_id
                self.predicted_collisions += 1
                self._log_event(track.track_id, f"Collision Prediction:{other.track_id}", "motion")
                return
        track.predicted_collision_seconds = None
        track.predicted_collision_target = None

    def _update_zone_state(self, track: AnalyticsTrack, current_time: datetime) -> None:
        zone_name = self._zone_for_point(track.center)
        if zone_name is None:
            if track.current_zone is not None:
                track.previous_zone = track.current_zone
                track.exit_time = current_time
                if track.entry_time is not None:
                    track.dwell_time += (current_time - track.entry_time).total_seconds()
                self._log_event(track.track_id, f"Exited Zone:{track.current_zone}", "zone")
            track.current_zone = None
            track.entry_time = None
            return

        if track.current_zone != zone_name:
            if track.current_zone is None:
                track.previous_zone = None
                track.entry_time = current_time
                track.exit_time = None
                self._log_event(track.track_id, f"Entered Zone:{zone_name}", "zone")
            else:
                track.previous_zone = track.current_zone
                track.exit_time = current_time
                if track.entry_time is not None:
                    track.dwell_time += (current_time - track.entry_time).total_seconds()
                self._log_event(track.track_id, f"Exited Zone:{track.current_zone}", "zone")
                track.entry_time = current_time
                self._log_event(track.track_id, f"Entered Zone:{zone_name}", "zone")
            track.current_zone = zone_name

    def _update_line_crossings(self, track: AnalyticsTrack, current_time: datetime) -> None:
        if not self.lines:
            return

        for line in self.lines:
            before = point_to_line_side(track.previous_position, line.start, line.end)
            after = point_to_line_side(track.center, line.start, line.end)
            if before == 0 or after == 0 or before * after > 0:
                continue
            self.line_counts["TOTAL"] += 1
            if after > 0:
                self.line_counts["IN"] += 1
            else:
                self.line_counts["OUT"] += 1
            self._log_event(track.track_id, f"Crossed Line:{line.name}", "line")

    def _zone_for_point(self, point: Tuple[int, int]) -> Optional[str]:
        for zone in self.zones:
            if point_in_polygon(point, zone.points):
                return zone.name
        return None

    def _log_event(self, track_id: int, event_type: str, source: str) -> None:
        self.events.append(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "track_id": track_id,
                "event_type": event_type,
                "source": source,
            }
        )
        if len(self.events) > 200:
            self.events = self.events[-200:]

    def draw(self, frame: np.ndarray) -> None:
        for line in self.lines:
            cv2.line(frame, line.start, line.end, (0, 0, 255), 2)
        for zone in self.zones:
            points = np.array(zone.points, dtype=np.int32)
            cv2.polylines(frame, [points], True, (0, 255, 255), 2)
        for track in self.tracks.values():
            if self.trails_enabled and len(track.trail) > 1:
                trail_points = np.array(track.trail, dtype=np.int32)
                cv2.polylines(frame, [trail_points], False, (255, 255, 255), 1)
            if track.center:
                cv2.circle(frame, (int(track.center[0]), int(track.center[1])), 3, (0, 255, 0), -1)
                if track.safe_distance_violation:
                    cv2.circle(frame, (int(track.center[0]), int(track.center[1])), 8, (0, 255, 255), 2)
                if track.predicted_collision_seconds is not None:
                    cv2.line(frame, (int(track.center[0]), int(track.center[1])), (int(track.center[0] + 30), int(track.center[1])), (0, 0, 255), 2)

    def get_summary(self) -> Dict[str, Any]:
        average_scene_speed = 0.0
        if self.tracks:
            average_scene_speed = sum(track.average_speed_mps for track in self.tracks.values()) / max(1, len(self.tracks))
        fastest = max(self.tracks.values(), key=lambda item: item.max_speed_mps, default=None)
        closest = None
        if len(self.tracks) > 1:
            pairs = []
            track_values = list(self.tracks.values())
            for index, left in enumerate(track_values):
                for right in track_values[index + 1:]:
                    if left.center and right.center:
                        pairs.append((distance_between(left.center, right.center), left.track_id, right.track_id))
            if pairs:
                closest = min(pairs, key=lambda item: item[0])
        return {
            "line_counts": dict(self.line_counts),
            "zone_counts": dict(self.zone_counts),
            "tracked_objects": len(self.tracks),
            "line_count": len(self.lines),
            "zone_count": len(self.zones),
            "events": len(self.events),
            "average_scene_speed": round(average_scene_speed, 3),
            "fastest_object": fastest.track_id if fastest else None,
            "closest_object_pair": closest,
            "safe_distance_violations": self.safe_distance_violations,
            "predicted_collisions": self.predicted_collisions,
        }


def distance_between(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def direction_from_delta(dx: int | float, dy: int | float) -> str:
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return ""
    if abs(dx) < 1e-6:
        return "N" if dy < 0 else "S"
    if abs(dy) < 1e-6:
        return "E" if dx > 0 else "W"
    if dx < 0 and dy > 0:
        return "NW"
    if dx > 0 and dy > 0:
        return "NE"
    if dx < 0 and dy < 0:
        return "SW"
    if dx > 0 and dy < 0:
        return "SE"
    return "E" if dx > 0 else "W"


def point_to_line_side(point: Tuple[int, int], line_start: Tuple[int, int], line_end: Tuple[int, int]) -> int:
    dx = line_end[0] - line_start[0]
    dy = line_end[1] - line_start[1]
    px = point[0] - line_start[0]
    py = point[1] - line_start[1]
    return int(np.sign(dx * py - dy * px))


def point_in_polygon(point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
    x, y = point
    inside = False
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside
