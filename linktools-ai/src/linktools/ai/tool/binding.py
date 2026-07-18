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


class ToolExecutionBindingFactory:
    def build(self, *, descriptor, arguments: Mapping[str, Any], context,
              result_processor_revision: str) -> ToolExecutionBinding:
        from ..errors import ToolBindingError
        metadata = context.metadata
        required = {"descriptor_fingerprint": descriptor.fingerprint(),
            "handler_revision": metadata.get("handler_revision"),
            "provider_revision": metadata.get("provider_revision"),
            "policy_revision": metadata.get("policy_revision"),
            "capability_revision": metadata.get("capability_revision")}
        missing = [key for key, value in required.items()
                   if not isinstance(value, str) or not value]
        if not isinstance(result_processor_revision, str) or not result_processor_revision:
            missing.append("result_processor_revision")
        if missing:
            raise ToolBindingError("tool execution binding is incomplete: " + ", ".join(missing))
        arguments_hash = hashlib.sha256(canonical_json(
            {"tool": descriptor.name, "arguments": dict(arguments)}
        ).encode("utf-8")).hexdigest()
        return ToolExecutionBinding(schema_version=1, tool_name=descriptor.name,
            arguments_hash=arguments_hash,
            descriptor_fingerprint=required["descriptor_fingerprint"],
            handler_revision=required["handler_revision"],
            provider_revision=required["provider_revision"],
            policy_revision=required["policy_revision"],
            capability_revision=required["capability_revision"],
            result_processor_revision=result_processor_revision)
