"""Canonical identity for an approval-able tool execution."""
from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

from ..json import canonical_json


@dataclass(frozen=True, slots=True)
class ToolExecutionBinding:
    schema_version: int
    tool_name: str
    arguments_hash: str
    descriptor_fingerprint: str
    handler_revision: str
    provider_revision: str
    policy_revision: str
    capability_revision: str
    result_processor_revision: str

    def __post_init__(self) -> None:
        for name in ("tool_name", "arguments_hash", "descriptor_fingerprint",
                     "handler_revision", "provider_revision", "policy_revision",
                     "capability_revision", "result_processor_revision"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"ToolExecutionBinding.{name} is required")

    def to_payload(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in (
            "schema_version", "tool_name", "arguments_hash",
            "descriptor_fingerprint", "handler_revision", "provider_revision",
            "policy_revision", "capability_revision", "result_processor_revision")}

    def fingerprint(self) -> str:
        return hashlib.sha256(canonical_json(self.to_payload()).encode()).hexdigest()

    def matches(self, other: Mapping[str, Any]) -> bool:
        return all(other.get(k) == v for k, v in self.to_payload().items())
