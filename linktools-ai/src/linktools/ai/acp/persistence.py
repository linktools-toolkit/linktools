#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ACP session sidecars and the LocalDirectoryStorage process lock."""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Mapping


@dataclass(frozen=True, slots=True)
class AcpSessionRecord:
    schema_version: "Literal[1]"
    session_id: str
    cwd: str
    additional_directories: "tuple[str, ...]"
    mode_id: str
    config_values: "Mapping[str, Any]"
    mcp_server_fingerprints: "tuple[str, ...]"
    title: "str | None"
    closed: bool
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported ACP session sidecar schema")


class AcpSessionRepository:
    def __init__(self, state_root: "str | Path") -> None:
        self.root = Path(state_root) / "acp" / "sessions"

    def path_for(self, session_id: str) -> Path:
        if not session_id or any(char in session_id for char in "/\\"):
            raise ValueError("invalid ACP session id")
        return self.root / f"{session_id}.json"

    def load(self, session_id: str) -> "AcpSessionRecord | None":
        path = self.path_for(session_id)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        if raw.get("schema_version") != 1:
            raise ValueError("unsupported ACP session sidecar schema")
        raw["additional_directories"] = tuple(raw.get("additional_directories", ()))
        raw["mcp_server_fingerprints"] = tuple(raw.get("mcp_server_fingerprints", ()))
        raw["created_at"] = datetime.fromisoformat(raw["created_at"])
        raw["updated_at"] = datetime.fromisoformat(raw["updated_at"])
        return AcpSessionRecord(**raw)

    def list(self) -> "tuple[AcpSessionRecord, ...]":
        if not self.root.is_dir():
            return ()
        records = []
        for path in self.root.glob("*.json"):
            record = self.load(path.stem)
            if record is not None:
                records.append(record)
        return tuple(records)

    def save(self, record: AcpSessionRecord) -> None:
        path = self.path_for(record.session_id)
        self.root.mkdir(parents=True, exist_ok=True)
        payload = asdict(record)
        payload["created_at"] = record.created_at.isoformat()
        payload["updated_at"] = record.updated_at.isoformat()
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=self.root)
        temp_path = Path(temp_name)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, path)
            directory_fd = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temp_path.unlink(missing_ok=True)


def mcp_descriptor_fingerprint(descriptor: object) -> str:
    if hasattr(descriptor, "model_dump"):
        raw = descriptor.model_dump(mode="json", by_alias=True, exclude_none=True)
    elif isinstance(descriptor, Mapping):
        raw = dict(descriptor)
    else:
        raw = {"value": str(descriptor)}
    raw = _without_secret_fields(raw)
    encoded = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _without_secret_fields(value: Any) -> Any:
    secret_names = {
        "authorization",
        "api_key",
        "apikey",
        "env",
        "environment",
        "headers",
        "password",
        "secret",
        "token",
    }
    if isinstance(value, Mapping):
        return {
            key: _without_secret_fields(item)
            for key, item in value.items()
            if str(key).lower().replace("-", "_") not in secret_names
        }
    if isinstance(value, (list, tuple)):
        return [_without_secret_fields(item) for item in value]
    return value


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
        stream.write(json.dumps({
            "pid": os.getpid(),
            "started_at": datetime.now().astimezone().isoformat(),
            "project_root": str(Path(project_root).resolve()),
            "sdk_version": self.sdk_version,
        }, sort_keys=True))
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


__all__ = ["AcpSessionRecord", "AcpSessionRepository", "ProjectProcessLock", "mcp_descriptor_fingerprint"]
