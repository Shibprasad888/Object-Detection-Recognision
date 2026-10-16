from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class SessionSummary:
    """Snapshot of important runtime statistics captured at shutdown."""

    session_id: str
    duration_seconds: float
    objects_detected: int
    objects_recognized: int
    alerts_generated: int
    frames_processed: int
    average_fps: float
    average_cpu_usage: float
    average_gpu_usage: float
    recording_duration_seconds: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "session_id": self.session_id,
            "duration_seconds": round(self.duration_seconds, 2),
            "objects_detected": self.objects_detected,
            "objects_recognized": self.objects_recognized,
            "alerts_generated": self.alerts_generated,
            "frames_processed": self.frames_processed,
            "average_fps": round(self.average_fps, 2),
            "average_cpu_usage": round(self.average_cpu_usage, 2),
            "average_gpu_usage": round(self.average_gpu_usage, 2),
            "recording_duration_seconds": round(self.recording_duration_seconds, 2),
        }


def write_session_summary(summary: SessionSummary, base_dir: Optional[Path] = None) -> Path:
    """Persist a session summary to disk in JSON format."""
    base = Path(base_dir or Path(__file__).resolve().parent)
    output_dir = base / "sessions"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{summary.session_id}.json"
    path.write_text(json.dumps(summary.to_dict(), indent=2))
    return path
