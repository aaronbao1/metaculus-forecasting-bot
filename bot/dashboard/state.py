# bot/dashboard/state.py
"""In-memory dashboard state with a background refresh thread."""
from __future__ import annotations

import copy
import threading
from datetime import datetime, timezone
from typing import Callable


class DashboardState:
    def __init__(self, collectors_map: dict[str, Callable[[], dict]], interval: float = 60.0):
        self._collectors = collectors_map
        self._interval = interval
        self._lock = threading.Lock()
        self._snapshot: dict = {"panels": {}, "generated_at": None}
        self._stop = threading.Event()

    def refresh_once(self) -> None:
        panels = {}
        for name, fn in self._collectors.items():
            try:
                panels[name] = fn()
            except Exception as exc:  # noqa: BLE001 - one dead source must not break others
                panels[name] = {"available": False, "error": str(exc)}
        snap = {"panels": panels,
                "generated_at": datetime.now(timezone.utc).isoformat()}
        with self._lock:
            self._snapshot = snap

    def get(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._snapshot)

    def run_forever(self) -> None:
        while not self._stop.is_set():
            self.refresh_once()
            self._stop.wait(self._interval)

    def start_background(self) -> threading.Thread:
        t = threading.Thread(target=self.run_forever, daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        self._stop.set()
