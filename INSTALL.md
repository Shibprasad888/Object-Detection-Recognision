# Installation Guide

## Requirements

- Python 3.11
- A compatible webcam or video file

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # macOS / Linux
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Packaging

```bash
pip install pyinstaller
pyinstaller --onefile --name object-detection main.py
```
