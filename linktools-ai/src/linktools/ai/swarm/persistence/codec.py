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


def _restore(value: object) -> object:
    if isinstance(value, list):
        return [_restore(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {"__decimal__"}:
            return Decimal(value["__decimal__"])
        if set(value) == {"__bytes__"}:
            return bytes.fromhex(value["__bytes__"])
        return {key: _restore(item) for key, item in value.items()}
    return value


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

    def decode_request(self, operation: str, payload: bytes) -> object:
        """Decode the durable command shape used by filesystem recovery.

        Start is the only operation that can be forward-recovered from the
        single-phase filesystem journal. Other operations fail closed until a
        complete multi-phase recovery protocol exists for them.
        """
        from datetime import datetime

        from ...events.context import EventStreamContext
        from ...events.payloads import SwarmStarted
        from ..commit import (
            StartSwarmCommand, StartSwarmPayload, SwarmCommitId,
            SwarmExecutionFence,
        )
        from ..models import SwarmRun, SwarmStatus, TokenUsage

        value = json.loads(payload)
        if value.get("schema_version") != self.schema_version or value.get("operation") != operation:
            raise ValueError("unsupported Swarm request payload")
        command = _restore(value.get("command"))
        if not isinstance(command, dict):
            raise ValueError("Swarm request command must be an object")
        identity = command.get("commit_id")
        swarm_run_id = command.get("swarm_run_id")
        if (
            not isinstance(identity, dict)
            or not isinstance(identity.get("value"), str)
            or not identity["value"]
            or not isinstance(swarm_run_id, str)
            or not swarm_run_id
        ):
            raise ValueError("Swarm request command has invalid identity")
        if operation != "start":
            # Recovery intentionally does not replay non-Start business writes,
            # but it must still decode and authenticate their journal identity
            # before failing closed.
            return _DecodedSwarmCommand(
                commit_id=identity["value"],
                swarm_run_id=swarm_run_id,
                encoded_payload=payload,
            )
        run = command.get("payload", {}).get("run", {})
        usage = run.get("token_usage", {})
        return StartSwarmCommand(
            commit_id=SwarmCommitId(command["commit_id"]["value"]),
            swarm_run_id=command["swarm_run_id"],
            expected_version=command["expected_version"],
            payload=StartSwarmPayload(
                run=SwarmRun(
                    id=run["id"], run_id=run["run_id"], round=run["round"],
                    status=SwarmStatus(run["status"]), version=run["version"],
                    token_usage=TokenUsage(
                        input_tokens=usage["input_tokens"],
                        output_tokens=usage["output_tokens"],
                        total_cost=usage["total_cost"],
                    ),
                    cost=run["cost"],
                    created_at=datetime.fromisoformat(run["created_at"]),
                    updated_at=datetime.fromisoformat(run["updated_at"]),
                    metadata=run["metadata"],
                    execution_token=run.get("execution_token"),
                ),
                started_event=SwarmStarted(**command["payload"]["started_event"]),
                event_context=EventStreamContext(**command["payload"]["event_context"]),
            ),
            fence=SwarmExecutionFence(command["fence"]["token"]),
        )

    def request_hash(self, operation: str, command: object) -> bytes:
        if isinstance(command, _DecodedSwarmCommand):
            return sha256(command.encoded_payload).digest()
        return sha256(self.encode_request(operation, command)).digest()


@dataclasses.dataclass(frozen=True, slots=True)
class _DecodedSwarmCommand:
    commit_id: str
    swarm_run_id: str
    encoded_payload: bytes


__all__ = ["SwarmCommitCodec"]
