from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Optional


class AppLogger:
    """Central logging utility with category-specific files and rotation."""

    def __init__(self, base_dir: Optional[Path] = None, level: str = "INFO") -> None:
        self.base_dir = Path(base_dir or Path(__file__).resolve().parent)
        self.log_dir = self.base_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.level = getattr(logging, level.upper(), logging.INFO)
        self._loggers: Dict[str, logging.Logger] = {}

    def get_logger(self, category: str) -> logging.Logger:
        if category in self._loggers:
            return self._loggers[category]

        logger = logging.getLogger(f"object_detection.{category}")
        logger.setLevel(self.level)
        logger.propagate = False

        if not logger.handlers:
            handler = RotatingFileHandler(self.log_dir / f"{category}.log", maxBytes=2 * 1024 * 1024, backupCount=3)
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
            logger.addHandler(handler)

        self._loggers[category] = logger
        return logger
