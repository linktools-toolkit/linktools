#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable references for named behavior and digest inputs for Agent bindings."""

from collections.abc import Mapping
from dataclasses import dataclass
from ..core import JsonValue, canonical_sha256
from ..errors import AIError, ErrorCode
from ._schema import canonicalize_json_schema

_CONTRIBUTION_KINDS = frozenset(
    {"agent", "tool", "skill", "mcp", "capability"}
)


@dataclass(frozen=True, slots=True)
class CapabilityRef:
    """Named capability identity without its behavior contract or locators."""

    kind: str
    id: str
    revision: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.kind, str)
            or self.kind not in _CONTRIBUTION_KINDS
            or not isinstance(self.id, str)
            or not self.id.strip()
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _positive_int(self.revision)

    def to_payload(self) -> "dict[str, JsonValue]":
        return {"kind": self.kind, "id": self.id, "revision": self.revision}

    @classmethod
    def from_payload(cls, value: object) -> "CapabilityRef":
        if not isinstance(value, Mapping) or not {"kind", "id", "revision"}.issubset(value):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return cls(value["kind"], value["id"], value["revision"])


def agent_ref_payload(
    contract: Mapping[str, JsonValue],
) -> "dict[str, JsonValue]":
    _require_format_v1(contract)
    return CapabilityRef("agent", _text(contract.get("id")), _revision(contract)).to_payload()


def capability_ref_payload(
    kind: str,
    identity: str,
    contract: Mapping[str, JsonValue],
) -> "dict[str, JsonValue]":
    if not isinstance(contract, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    _require_format_v1(contract)
    reference = CapabilityRef(kind, identity, _revision(contract))
    if reference.kind in {"agent", "skill", "mcp"} and contract.get("id") != reference.id:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return reference.to_payload()


def binding_digest_payload(
    payload: Mapping[str, JsonValue],
) -> "dict[str, JsonValue]":
    if not isinstance(payload, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    required = {
        "version",
        "agent_spec",
        "model_contract",
        "selected",
        "subagents",
        "output_mode",
        "output_schema",
    }
    if not required.issubset(payload):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    _require_format_v1(payload)
    agent_spec = _mapping(payload["agent_spec"])
    _mapping(payload["model_contract"])  # Diagnostic metadata, not model identity.
    selected = payload["selected"]
    subagents = payload["subagents"]
    if not isinstance(selected, list) or not isinstance(subagents, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    selected_refs: list[JsonValue] = [
        CapabilityRef.from_payload(item).to_payload() for item in selected
    ]

    subagent_refs: list[JsonValue] = []
    for item in subagents:
        value = _mapping(item)
        if value.get("kind") != "agent":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        subagent_refs.append(
            CapabilityRef(
                "agent",
                _text(value.get("id")),
                _positive_int(value.get("revision", 1)),
            ).to_payload()
        )

    output_mode = payload["output_mode"]
    output_schema = payload["output_schema"]
    if output_mode not in {"text", "structured"} or not isinstance(
        output_schema, Mapping
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    result: dict[str, JsonValue] = {
        "contract": "agent-binding-v1",
        "agent": agent_ref_payload(agent_spec),
        "model": _text(agent_spec.get("model")),
        "selected": selected_refs,
        "subagents": subagent_refs,
        "output": {
            "mode": output_mode,
            "schema": canonicalize_json_schema(output_schema),
        },
    }
    children = payload.get("subagent_bindings")
    if children is not None:
        if not isinstance(children, list):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        child_refs: list[dict[str, JsonValue]] = []
        for child in children:
            child_payload = _mapping(child)
            child_spec = _mapping(child_payload.get("agent_spec"))
            child_refs.append(
                {
                    "agent": agent_ref_payload(child_spec),
                    "binding_digest": canonical_sha256(
                        binding_digest_payload(child_payload)
                    ),
                }
            )
        child_refs.sort(key=lambda value: value["agent"]["id"])
        result["subagent_bindings"] = child_refs
    return result


def _require_format_v1(value: Mapping[str, JsonValue]) -> None:
    version = value.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if version != 1:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)


def _revision(value: Mapping[str, JsonValue]) -> int:
    return _positive_int(value.get("revision", 1))


def _positive_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def _text(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def _mapping(value: object) -> "dict[str, JsonValue]":
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return dict(value)


__all__ = [
    "CapabilityRef",
    "agent_ref_payload",
    "binding_digest_payload",
    "capability_ref_payload",
]
