from __future__ import annotations

import inspect
from pathlib import Path

from app_runtime import SessionSummary, write_session_summary
from config_manager import ConfigManager
from detector import ObjectDetector
from logging_manager import AppLogger


def test_object_detector_accepts_device_argument() -> None:
    signature = inspect.signature(ObjectDetector.__init__)
    assert "device" in signature.parameters


def test_config_manager_persists_nested_settings(tmp_path: Path) -> None:
    config = ConfigManager(base_dir=tmp_path)
    config.set("app.version", "1.0.0")
    config.set("logging.level", "INFO")
    config.set("output_paths.recordings", str(tmp_path / "recordings"))

    reloaded = ConfigManager(base_dir=tmp_path)

    assert reloaded.get("app.version") == "1.0.0"
    assert reloaded.get("logging.level") == "INFO"
    assert reloaded.get("output_paths.recordings") == str(tmp_path / "recordings")


def test_logger_creates_category_specific_log_files(tmp_path: Path) -> None:
    logger_manager = AppLogger(base_dir=tmp_path)
    app_logger = logger_manager.get_logger("app")
    app_logger.info("startup ready")

    log_file = tmp_path / "logs" / "app.log"
    assert log_file.exists()


def test_session_summary_writes_json(tmp_path: Path) -> None:
    summary = SessionSummary(
        session_id="demo-session",
        duration_seconds=42.5,
        objects_detected=120,
        objects_recognized=18,
        alerts_generated=3,
        frames_processed=1500,
        average_fps=22.5,
        average_cpu_usage=35.0,
        average_gpu_usage=0.0,
        recording_duration_seconds=10.0,
    )
    path = write_session_summary(summary, tmp_path)

    assert path.exists()
    assert "demo-session" in path.read_text()
