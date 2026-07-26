"""Canonical codec shared by Filesystem and SQL Swarm coordinators."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from enum import Enum
from hashlib import sha256
from linktools.ai.json import canonical_json


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if isinstance(value, Decimal):
        return {"__decimal__": str(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    raise TypeError(f"cannot serialize {type(value)!r}")


class SwarmCommitCodec:
    schema_version = 1

    def encode_request(self, operation: str, command: object) -> bytes:
        if dataclasses.is_dataclass(command):
            value = _jsonable(command)
        elif isinstance(command, dict):
            value = command
        else:
            value = {"command": str(command)}
        return canonical_json({"schema_version": self.schema_version, "operation": operation, "command": value}).encode()

    def encode_result(self, operation: str, result: object) -> bytes:
        return canonical_json({"schema_version": self.schema_version, "operation": operation, "result": _jsonable(result)}).encode()

    def decode_result(self, operation: str, payload: bytes) -> object:
        value = json.loads(payload)
        if value.get("schema_version") != self.schema_version or value.get("operation") != operation:
            raise ValueError("unsupported Swarm result payload")
        return value["result"]

    def request_hash(self, operation: str, command: object) -> bytes:
        return sha256(self.encode_request(operation, command)).digest()


__all__ = ["SwarmCommitCodec"]
