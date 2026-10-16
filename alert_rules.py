from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from scene_analytics import AnalyticsTrack, SceneAnalytics
from alert_manager import AlertManager


class AlertRules:
    """Rule-based alert evaluation on top of the existing analytics state."""

    def __init__(self, scene_analytics: SceneAnalytics, alert_manager: AlertManager) -> None:
        self.scene_analytics = scene_analytics
        self.alert_manager = alert_manager
        self.restricted_zones: List[str] = []
        self.allowed_zones: Dict[int, List[str]] = {}
        self.speed_limits: Dict[str, float] = {}
        self.loitering_threshold_seconds = 30.0
        self.unknown_object_timeout_seconds = 10.0
        self.stationary_duration_seconds = 8.0
        self.missing_object_timeout_seconds = 15.0
        self._last_alerts: Dict[Tuple[str, Optional[int], Optional[str]], datetime] = {}

    def evaluate(self, frame: Any, annotated_frame: Any) -> None:
        for track in list(self.scene_analytics.tracks.values()):
            self._check_restricted_zone(track)
            self._check_loitering(track)
            self._check_safe_distance(track)
            self._check_speed_violation(track)
            self._check_collision_warning(track)
            self._check_wrong_zone(track)
            self._check_unknown_object(track)
            self._check_missing_object(track)

    def _check_restricted_zone(self, track: AnalyticsTrack) -> None:
        if not track.current_zone or track.current_zone not in self.restricted_zones:
            return
        self._emit_alert(
            "Restricted Zone Intrusion",
            "WARNING",
            f"{track.object_name} entered restricted zone {track.current_zone}",
            track.track_id,
            track.object_name,
            track.class_name,
            track.current_zone,
        )

    def _check_loitering(self, track: AnalyticsTrack) -> None:
        if not track.current_zone or track.entry_time is None:
            return
        dwell_seconds = (datetime.now() - track.entry_time).total_seconds()
        if dwell_seconds >= self.loitering_threshold_seconds:
            self._emit_alert(
                "Loitering Detection",
                "WARNING",
                f"{track.object_name} has remained in {track.current_zone} for {int(dwell_seconds)}s",
                track.track_id,
                track.object_name,
                track.class_name,
                track.current_zone,
            )

    def _check_safe_distance(self, track: AnalyticsTrack) -> None:
        if track.safe_distance_violation:
            self._emit_alert(
                "Safe Distance Violation",
                "WARNING",
                f"{track.object_name} violated the minimum safe distance",
                track.track_id,
                track.object_name,
                track.class_name,
                track.current_zone,
            )

    def _check_speed_violation(self, track: AnalyticsTrack) -> None:
        speed_limit = self.speed_limits.get(track.class_name.lower())
        if speed_limit is None or track.current_speed_mps <= speed_limit:
            return
        self._emit_alert(
            "Speed Violation",
            "WARNING",
            f"{track.object_name} exceeded speed limit ({track.current_speed_mps:.2f} m/s > {speed_limit:.2f} m/s)",
            track.track_id,
            track.object_name,
            track.class_name,
            track.current_zone,
            metadata={"speed": track.current_speed_mps, "speed_limit": speed_limit},
        )

    def _check_collision_warning(self, track: AnalyticsTrack) -> None:
        if track.predicted_collision_seconds is None:
            return
        self._emit_alert(
            "Collision Warning",
            "CRITICAL",
            f"Possible collision in {track.predicted_collision_seconds:.1f}s",
            track.track_id,
            track.object_name,
            track.class_name,
            track.current_zone,
            metadata={"predicted_collision_seconds": track.predicted_collision_seconds, "target_track_id": track.predicted_collision_target},
        )

    def _check_wrong_zone(self, track: AnalyticsTrack) -> None:
        allowed = self.allowed_zones.get(track.track_id)
        if not allowed or not track.current_zone:
            return
        if track.current_zone not in allowed:
            self._emit_alert(
                "Wrong Zone Detection",
                "WARNING",
                f"{track.object_name} entered disallowed zone {track.current_zone}",
                track.track_id,
                track.object_name,
                track.class_name,
                track.current_zone,
            )

    def _check_unknown_object(self, track: AnalyticsTrack) -> None:
        if track.recognition_status == "Known":
            return
        if track.last_seen is None:
            return
        age_seconds = (datetime.now() - track.last_seen).total_seconds()
        if age_seconds >= self.unknown_object_timeout_seconds:
            self._emit_alert(
                "Unknown Object Alert",
                "INFO",
                f"Unknown object {track.track_id} has remained visible for {int(age_seconds)}s",
                track.track_id,
                track.object_name,
                track.class_name,
                track.current_zone,
            )

    def _check_missing_object(self, track: AnalyticsTrack) -> None:
        if track.recognition_status != "Known" or track.last_seen is None:
            return
        age_seconds = (datetime.now() - track.last_seen).total_seconds()
        if age_seconds >= self.missing_object_timeout_seconds:
            self._emit_alert(
                "Missing Object Detection",
                "WARNING",
                f"Registered object {track.object_name} is missing",
                track.track_id,
                track.object_name,
                track.class_name,
                track.current_zone,
            )

    def _emit_alert(
        self,
        alert_type: str,
        severity: str,
        description: str,
        track_id: Optional[int],
        object_name: Optional[str],
        object_class: Optional[str],
        zone: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        key = (alert_type, track_id, zone)
        now = datetime.now()
        last_time = self._last_alerts.get(key)
        cooldown_seconds = 3.0 if severity == "INFO" else 6.0
        if last_time is not None and (now - last_time).total_seconds() < cooldown_seconds:
            return
        self._last_alerts[key] = now
        self.alert_manager.create_alert(
            alert_type=alert_type,
            severity=severity,
            description=description,
            track_id=track_id,
            object_name=object_name,
            object_class=object_class,
            zone=zone,
            metadata=metadata,
        )
