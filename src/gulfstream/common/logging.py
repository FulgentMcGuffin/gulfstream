"""Logging context manager for pipeline runs."""
from __future__ import annotations

import logging
import os
from datetime import datetime


class LoggingContext:
    """Configure root logging to a timestamped file under log_dir."""

    def __init__(self, log_dir: str, log_level: str = "INFO") -> None:
        self.log_dir = log_dir
        self.log_level = log_level.upper()
        self._handler: logging.Handler | None = None
        self._prev_level: int | None = None

    def __enter__(self) -> LoggingContext:
        os.makedirs(self.log_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.log_dir, f"gulfstream_{stamp}.log")
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        root = logging.getLogger()
        self._prev_level = root.level
        root.setLevel(getattr(logging, self.log_level, logging.INFO))
        root.addHandler(handler)
        # Also ensure a console handler exists.
        if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
                    for h in root.handlers):
            console = logging.StreamHandler()
            console.setFormatter(
                logging.Formatter("%(levelname)s [%(name)s] %(message)s")
            )
            root.addHandler(console)
        self._handler = handler
        logging.getLogger(__name__).info("Logging to %s", path)
        return self

    def __exit__(self, *exc) -> None:
        root = logging.getLogger()
        if self._handler is not None:
            root.removeHandler(self._handler)
            self._handler.close()
            self._handler = None
        if self._prev_level is not None:
            root.setLevel(self._prev_level)
