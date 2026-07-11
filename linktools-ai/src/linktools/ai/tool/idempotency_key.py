#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IdempotencyKeyBuilder: derives a stable idempotency key for a tool call.

The key must NOT be ``tool_name + call_id`` alone -- the same logical call gets
a different model ``call_id`` on retry/resume, so keying on it would defeat
replay. The default builder hashes the run, the tool, a canonical dump of the
final arguments, and the schema_version, so:
  - the same call replays to the cached result,
  - the same key with different arguments is a conflict,
  - a schema_version change is a fresh idempotency record.

An explicit business key (declared via tool metadata/descriptor and explicitly
ALLOWED by policy) takes precedence -- but it is never trusted just because the
model passed it."""

import hashlib
import json
from typing import TYPE_CHECKING, Any, Mapping, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..run.context import RunContext
    from ..security.descriptor import ToolDescriptor


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


@runtime_checkable
class IdempotencyKeyBuilder(Protocol):
    def build(
        self,
        *,
        descriptor: "ToolDescriptor",
        arguments: "Mapping[str, Any]",
        run_context: "RunContext | None",
        schema_version: str,
    ) -> "str | None":
        ...


class DefaultIdempotencyKeyBuilder:
    """sha256(run_id + tool_name + canonical_json(arguments) + schema_version).
    Returns None when there is no run_id (the call is not part of a persisted
    run, so idempotent replay is meaningless)."""

    def build(
        self,
        *,
        descriptor: "ToolDescriptor",
        arguments: "Mapping[str, Any]",
        run_context: "RunContext | None",
        schema_version: str,
    ) -> "str | None":
        run_id = getattr(run_context, "run_id", None) if run_context else None
        if not run_id:
            return None
        payload = "|".join([
            run_id, descriptor.name, _canonical_json(dict(arguments)), schema_version,
        ]).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
