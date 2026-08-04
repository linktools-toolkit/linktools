#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ACP session sidecar persistence."""

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
        staged = self.stage(record)
        try:
            self.publish(staged, record.session_id)
        finally:
            self.discard(staged)

    def stage(self, record: AcpSessionRecord) -> Path:
        """Write a durable sidecar temporary file for a multi-system commit."""
        path = self.path_for(record.session_id)
        self.root.mkdir(parents=True, exist_ok=True)
        payload = asdict(record)
        payload["created_at"] = record.created_at.isoformat()
        payload["updated_at"] = record.updated_at.isoformat()
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=self.root)
        temp_path = Path(temp_name)
        staged = False
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            staged = True
            return temp_path
        finally:
            if temp_path.exists() and not staged:
                temp_path.unlink(missing_ok=True)

    def publish(self, staged: Path, session_id: str) -> None:
        path = self.path_for(session_id)
        os.replace(staged, path)
        directory_fd = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    @staticmethod
    def discard(staged: Path) -> None:
        staged.unlink(missing_ok=True)


__all__ = ["AcpSessionRecord", "AcpSessionRepository"]
