# bot/dashboard/state.py
"""In-memory dashboard state with a per-collector background refresh.

Collectors are ``{name: (fn, interval_seconds)}`` (a bare ``fn`` defaults to
``default_interval``). The loop ticks every ``tick`` seconds and re-runs a
collector only when its own interval has elapsed — so expensive sources (Metaculus)
poll far less often than cheap ones.
"""
from __future__ import annotations

import copy
import threading
import time
from datetime import datetime, timezone


class DashboardState:
    def __init__(self, collectors: dict, tick: float = 15.0,
                 default_interval: float = 60.0):
        self._collectors: dict[str, tuple] = {}
        for name, spec in collectors.items():
            fn, interval = spec if isinstance(spec, tuple) else (spec, default_interval)
            self._collectors[name] = (fn, interval)
        self._tick = tick
        self._last_run: dict[str, float | None] = {n: None for n in self._collectors}
        self._lock = threading.Lock()
        self._snapshot: dict = {"panels": {}, "generated_at": None}
        self._stop = threading.Event()

    def _run(self, name: str) -> dict:
        fn, _ = self._collectors[name]
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - one dead source must not break others
            return {"available": False, "error": str(exc)}

    def _commit(self, panels: dict) -> None:
        snap = {"panels": panels,
                "generated_at": datetime.now(timezone.utc).isoformat()}
        with self._lock:
            self._snapshot = snap

    def refresh_once(self) -> None:
        """Force-run every collector (startup)."""
        now = time.monotonic()
        panels = {}
        for name in self._collectors:
            panels[name] = self._run(name)
            self._last_run[name] = now
        self._commit(panels)

    def refresh_due(self) -> None:
        """Run only collectors whose interval has elapsed."""
        now = time.monotonic()
        with self._lock:
            panels = dict(self._snapshot["panels"])
        changed = False
        for name, (_, interval) in self._collectors.items():
            last = self._last_run[name]
            if last is None or (now - last) >= interval:
                panels[name] = self._run(name)
                self._last_run[name] = now
                changed = True
        if changed:
            self._commit(panels)

    def get(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._snapshot)

    def run_forever(self) -> None:
        while not self._stop.is_set():
            self.refresh_due()
            self._stop.wait(self._tick)

    def start_background(self) -> threading.Thread:
        t = threading.Thread(target=self.run_forever, daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        self._stop.set()
