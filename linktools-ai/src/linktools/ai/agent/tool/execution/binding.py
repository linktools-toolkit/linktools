"""Stable identity for approval and idempotent tool execution."""

from dataclasses import asdict, dataclass
from hashlib import sha256

from ....json import canonical_json_bytes, normalize_json


@dataclass(frozen=True, slots=True)
class ToolRevisionSet:
    descriptor: str
    handler: str
    provider: str
    policy: str
    feature: str
    result_processor: str


@dataclass(frozen=True, slots=True)
class ToolExecutionBinding:
    schema_version: int
    tool_name: str
    arguments_hash: str
    revisions: ToolRevisionSet

    def fingerprint(self) -> str:
        return sha256(
            canonical_json_bytes(normalize_json(asdict(self)))
        ).hexdigest()
