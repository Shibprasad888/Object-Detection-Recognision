# Object Detection & Recognition

A Python desktop application for real-time video analytics using OpenCV, YOLOv8, and CLIP-based object recognition. It detects and tracks objects from a webcam, supports registering objects, and provides scene analytics, alerts, and reports.

## Features

- Real-time object detection and tracking with YOLOv8
- Register, recognize, search, and review tracked objects
- Draw scene lines and zones; view motion trails and analytics
- Generate alerts and dashboard data, including heatmap and report exports
- Store object registry, alerts, and dashboard data in SQLite
- Configure the inference device and application settings

## Requirements

- Python 3.11
- A webcam (or another OpenCV-compatible camera source)
- macOS, Windows, or Linux with a desktop environment for the OpenCV display window

The repository includes the YOLOv8 Nano model at `models/yolov8n.pt`.

## Installation

Clone the repository and enter the project directory, then create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

On Windows, activate it with:

```powershell
.venv\Scripts\activate
```

Install the Python dependencies and start the application:

```bash
python -m pip install -r requirements.txt
python main.py
```

The application opens the default camera (`0`). Use `app_settings.json` and the configuration manager to adjust application settings, including the inference device.

## Controls

Click the video window where prompted to select objects or draw analytics regions. Some actions also prompt for input in the terminal.

| Key | Action |
| --- | --- |
| `Q` | Quit |
| `R` | Register a detected object |
| `E` / `D` | Edit / delete a registered object |
| `S` / `H` | Search objects / view object history |
| `L` / `Z` | Draw a line / zone |
| `T` | Toggle motion trails |
| `C` | Reset counters |
| `P` | Pause or resume analytics |
| `A` | Toggle alert sounds |
| `I` / `O` | Toggle / reset the heatmap |
| `U` | Export a daily report |
| `Esc` | Cancel line or zone drawing |

## Run the tests

Install the test dependency if needed, then run:

```bash
python -m pip install pytest
python -m pytest
```

## Project structure

- `main.py` — desktop application and keyboard controls
- `detector.py` — detection, tracking, registration, and recognition
- `scene_analytics.py` — zones, lines, trails, and motion analytics
- `alert_manager.py`, `alert_rules.py` — alert storage and rule evaluation
- `dashboard_manager.py` — dashboard data and report exports
- `camera_manager.py` — camera lifecycle management
- `config_manager.py` — application settings
- `models/` — YOLO model and runtime SQLite data
- `tests/` — regression tests

See [INSTALL.md](INSTALL.md) for installation notes, [USER_MANUAL.md](USER_MANUAL.md) for additional usage guidance, and [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for architecture details.
