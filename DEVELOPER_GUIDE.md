# Developer Guide

## Architecture Overview

The application is split into focused modules:

- `detector.py` holds the inference pipeline.
- `scene_analytics.py` handles movement, zones, and lines.
- `alert_manager.py` stores and raises alerts.
- `dashboard_manager.py` maintains analytical snapshots.
- `config_manager.py` stores and persists project settings.
- `logging_manager.py` handles structured logs.
- `app_runtime.py` writes session summaries.
- `camera_manager.py` manages camera streams.

## Extension Points

- Add new analytics rules in `alert_rules.py`.
- Add new UI overlays in `main.py`.
- Introduce new camera sources through `camera_manager.py`.
- Persist new settings in `config_manager.py`.
