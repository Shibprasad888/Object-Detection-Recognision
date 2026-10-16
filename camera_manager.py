from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class CameraStream:
    """Represents a single camera source and its processing state."""

    def __init__(self, camera_config: Dict[str, Any], detector_factory: Any) -> None:
        self.config = camera_config
        self.detector_factory = detector_factory
        self.name = camera_config.get("name", "Camera")
        self.source = camera_config.get("source", "0")
        self.device = camera_config.get("device", "cpu")
        self.running = False
        self.paused = False
        self.frame: Optional[np.ndarray] = None
        self.annotated_frame: Optional[np.ndarray] = None
        self.last_error: Optional[str] = None
        self.fps = 0.0
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.detector = None
        self.last_seen = datetime.now()

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def _run(self) -> None:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.last_error = f"Unable to open {self.source}"
            self.running = False
            return

        prev_time = time.time()
        while self.running:
            if self.paused:
                time.sleep(0.01)
                continue
            success, frame = cap.read()
            if not success or frame is None:
                self.last_error = f"Read failed for {self.source}"
                break
            with self.lock:
                self.frame = frame
            if self.detector is None:
                self.detector = self.detector_factory(self.device)
            try:
                annotated, _ = self.detector.detect(frame)
            except Exception as exc:
                self.last_error = str(exc)
                continue
            with self.lock:
                self.annotated_frame = annotated
            current_time = time.time()
            elapsed = max(1e-6, current_time - prev_time)
            self.fps = 1.0 / elapsed
            prev_time = current_time
            self.last_seen = datetime.now()
        cap.release()
        self.running = False

    def snapshot(self) -> Optional[np.ndarray]:
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            return None if self.annotated_frame is None else self.annotated_frame.copy()


class CameraManager:
    """Manages multiple camera streams and their lifecycle."""

    def __init__(self, detector_factory: Any, config_manager: Any) -> None:
        self.detector_factory = detector_factory
        self.config_manager = config_manager
        self.streams: List[CameraStream] = []

    def add_camera(self, camera_config: Dict[str, Any]) -> CameraStream:
        stream = CameraStream(camera_config, self.detector_factory)
        self.streams.append(stream)
        stream.start()
        self.config_manager.add_camera(camera_config)
        return stream

    def remove_camera(self, stream: CameraStream) -> None:
        stream.stop()
        self.streams = [item for item in self.streams if item is not stream]

    def get_active_streams(self) -> List[CameraStream]:
        return list(self.streams)
