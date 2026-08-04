#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Process-wide project lock for ACP stdio servers."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


class ProjectProcessLock:
    def __init__(self, path: "str | Path", *, sdk_version: str = "0.12.0") -> None:
        self.path = Path(path)
        self.sdk_version = sdk_version
        self._stream = None

    def acquire(self, *, project_root: "str | Path") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+", encoding="utf-8")
        try:
            self._lock_stream(stream)
        except OSError as exc:
            stream.seek(0)
            holder = stream.read().strip() or "unknown"
            stream.close()
            raise RuntimeError(f"ACP project lock is held: {holder}") from exc
        stream.seek(0)
        stream.truncate()
        stream.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "started_at": datetime.now().astimezone().isoformat(),
                    "project_root": str(Path(project_root).resolve()),
                    "sdk_version": self.sdk_version,
                },
                sort_keys=True,
            )
        )
        stream.flush()
        os.fsync(stream.fileno())
        self._stream = stream

    def release(self) -> None:
        if self._stream is None:
            return
        stream, self._stream = self._stream, None
        try:
            self._unlock_stream(stream)
        finally:
            stream.close()

    def _lock_stream(self, stream: Any) -> None:
        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock_stream(self, stream: Any) -> None:
        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


__all__ = ["ProjectProcessLock"]
