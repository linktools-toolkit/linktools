"""Single canonical wire codec for Run durable commit requests/results."""

from __future__ import annotations

import base64
import dataclasses
import json
from collections.abc import Mapping
from datetime import date, datetime, timezone
from enum import Enum
from hashlib import sha256
from typing import Any


class RunCommitCodec:
    schema_version = 1

    def _value(self, value: Any) -> Any:
        if dataclasses.is_dataclass(value):
            return {f.name: self._value(getattr(value, f.name)) for f in dataclasses.fields(value)}
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, bytes):
            return {"__bytes__": base64.b64encode(value).decode("ascii")}
        if isinstance(value, datetime):
            return {"__datetime__": value.astimezone(timezone.utc).isoformat()}
        if isinstance(value, date):
            return {"__date__": value.isoformat()}
        if isinstance(value, tuple):
            return [self._value(item) for item in value]
        if isinstance(value, dict):
            return {str(k): self._value(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
        if isinstance(value, Mapping):
            return {str(k): self._value(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
        if isinstance(value, (list, set, frozenset)):
            return [self._value(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return self._value(model_dump(mode="python"))
        if hasattr(value, "__dict__") and not value.__class__.__module__.startswith("builtins"):
            return self._value(vars(value))
        raise TypeError(f"unsupported durable commit value: {type(value)!r}")

    def _bytes(self, operation: Any, command: Any) -> bytes:
        return json.dumps({"schema_version": self.schema_version, "operation": str(operation), "command": self._value(command)}, separators=(",", ":"), ensure_ascii=False, sort_keys=True).encode()

    def encode_request(self, operation: Any, command: Any) -> bytes:
        return self._bytes(operation, command)

    def encode_result(self, operation: Any, result: Any) -> bytes:
        return self._bytes(operation, result)

    def decode_result(self, operation: Any, payload: bytes) -> Any:
        data = json.loads(payload)
        if data.get("schema_version") != self.schema_version or data.get("operation") != str(operation):
            raise ValueError("unsupported Run commit result payload")
        return data["command"]

    def request_hash(self, operation: Any, command: Any) -> bytes:
        return sha256(self.encode_request(operation, command)).digest()

    def result_hash(self, operation: Any, result: Any) -> bytes:
        return sha256(self.encode_result(operation, result)).digest()


__all__ = ["RunCommitCodec"]
