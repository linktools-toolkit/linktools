#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal semantic projections for capability and Agent binding identities."""

from collections.abc import Mapping
from typing import cast

from ..core import JsonValue, canonical_sha256
from ..errors import AIError, ErrorCode
from ._schema import canonicalize_json_schema

_CONTRIBUTION_KINDS = frozenset(
    {"agent", "tool", "skill", "mcp", "capability", "task", "task_expander"}
)
_AGENT_SPEC_FIELDS = (
    "version",
    "id",
    "model",
    "system_prompt",
    "instructions",
    "allow_tools",
    "allow_skills",
    "allow_subagents",
    "allow_capabilities",
    "usage_limits",
    "planning",
    "thinking",
    "tool_retries",
    "output_retries",
)


def capability_identity_payload(
    kind: str,
    identity: str,
    contract: Mapping[str, JsonValue],
) -> "dict[str, JsonValue]":
    """Return the one semantic projection for a capability contribution."""
    if kind not in _CONTRIBUTION_KINDS or not isinstance(identity, str) or not identity:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if not isinstance(contract, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    semantic = _capability_semantic(kind, contract)
    return {
        "contract": "capability-fingerprint-v1",
        "kind": kind,
        "id": identity,
        "semantic": semantic,
    }


def binding_identity_payload(
    payload: Mapping[str, JsonValue],
) -> "dict[str, JsonValue]":
    """Return the minimal identity projection for an Agent binding snapshot."""
    if not isinstance(payload, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    required = {
        "version",
        "agent_spec",
        "base_model",
        "selected",
        "subagents",
        "output_mode",
        "output_schema",
    }
    if not required.issubset(payload):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if (
        isinstance(payload["version"], bool)
        or not isinstance(payload["version"], int)
        or payload["version"] != 1
    ):
        if isinstance(payload["version"], int) and not isinstance(
            payload["version"], bool
        ):
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    agent_spec = _mapping(payload["agent_spec"])
    base_model = _mapping(payload["base_model"])
    selected = payload["selected"]
    subagents = payload["subagents"]
    if not isinstance(selected, list) or not isinstance(subagents, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    selected_projection: list[JsonValue] = []
    for item in selected:
        selected_value = _mapping(item)
        kind = selected_value.get("kind")
        identity = selected_value.get("id")
        contract = selected_value.get("contract")
        if (
            not isinstance(kind, str)
            or not isinstance(identity, str)
            or not isinstance(contract, Mapping)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        fingerprint = canonical_sha256(
            capability_identity_payload(kind, identity, cast(Mapping[str, JsonValue], contract))
        )
        selected_projection.append(
            {"kind": kind, "id": identity, "fingerprint": fingerprint}
        )

    subagent_projection: list[JsonValue] = []
    for item in subagents:
        value = _mapping(item)
        kind = value.get("kind")
        identity = value.get("id")
        if kind != "agent" or not isinstance(identity, str) or not identity:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        child: dict[str, JsonValue] = {"kind": kind, "id": identity}
        if "description" in value:
            description = value["description"]
            if description is not None and not isinstance(description, str):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            child["description"] = description
        subagent_projection.append(child)

    output_mode = payload["output_mode"]
    output_schema = payload["output_schema"]
    if output_mode not in {"text", "structured"} or not isinstance(output_schema, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    result: dict[str, JsonValue] = {
        "contract": "agent-binding-v1",
        "agent_spec": _agent_spec_semantic(agent_spec),
        "base_model": dict(base_model),
        "selected": selected_projection,
        "subagents": subagent_projection,
        "output_mode": output_mode,
        "output_schema": canonicalize_json_schema(
            cast(Mapping[str, JsonValue], output_schema)
        ),
    }
    children = payload.get("subagent_bindings")
    if children is not None:
        if not isinstance(children, list):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        child_projection: list[dict[str, JsonValue]] = []
        for child in children:
            child_payload = _mapping(child)
            child_spec = _mapping(child_payload.get("agent_spec"))
            child_id = child_spec.get("id")
            if not isinstance(child_id, str) or not child_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            child_projection.append(
                {
                    "id": child_id,
                    "binding_digest": canonical_sha256(
                        binding_identity_payload(child_payload)
                    ),
                }
            )
        child_projection.sort(key=lambda value: cast(str, value["id"]))
        result["subagent_bindings"] = child_projection
    return result


def _capability_semantic(
    kind: str,
    contract: Mapping[str, JsonValue],
) -> "dict[str, JsonValue]":
    version = contract.get("version")
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != 1
    ):
        if isinstance(version, int) and not isinstance(version, bool):
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if kind == "agent":
        return _agent_spec_semantic(contract)
    if kind == "tool":
        return _tool_semantic(contract)
    if kind == "skill":
        return _skill_semantic(contract)
    if kind == "mcp":
        return _fields(contract, ("version", "id", "command", "args"))
    if kind == "capability":
        return _fields(contract, ("version", "revision", "defer_loading", "config"))
    if kind == "task":
        return _task_semantic(contract)
    return _fields(contract, ("version", "expander_id", "expander_version"))


def _agent_spec_semantic(contract: Mapping[str, JsonValue]) -> "dict[str, JsonValue]":
    result = _fields(contract, _AGENT_SPEC_FIELDS)
    if "preload_skills" in contract:
        preload_skills = contract["preload_skills"]
        if not isinstance(preload_skills, list) or any(
            not isinstance(item, str) for item in preload_skills
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if preload_skills:
            result["preload_skills"] = preload_skills
    return result


def _tool_semantic(contract: Mapping[str, JsonValue]) -> "dict[str, JsonValue]":
    required = {"version", "description", "parameters", "return_schema", "strict", "metadata"}
    if not required.issubset(contract):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    metadata = _mapping(contract["metadata"])
    metadata.pop("linktools.tool_metrics_managed", None)
    return {
        "version": 1,
        "description": contract["description"],
        "parameters": canonicalize_json_schema(
            _mapping(contract["parameters"])
        ),
        "return_schema": canonicalize_json_schema(
            _mapping(contract["return_schema"])
        ),
        "strict": contract["strict"],
        "metadata": dict(metadata),
        **(
            {"semantic_revision": contract["semantic_revision"]}
            if "semantic_revision" in contract
            else {}
        ),
    }


def _skill_semantic(contract: Mapping[str, JsonValue]) -> "dict[str, JsonValue]":
    result = _fields(contract, ("version", "id", "content"))
    if "description" in contract:
        result["description"] = contract["description"]
    source = contract.get("source")
    if source is None:
        return result
    source_value = _mapping(source)
    source_projection = _fields(source_value, ("source_id", "root"))
    snapshot = source_value.get("snapshot")
    if snapshot is not None:
        snapshot_value = _mapping(snapshot)
        source_projection["snapshot"] = _fields(snapshot_value, ("digest", "size"))
    result["source"] = source_projection
    return result


def _task_semantic(contract: Mapping[str, JsonValue]) -> "dict[str, JsonValue]":
    result = _fields(contract, ("version", "task_type", "task_version", "effect"))
    result["reconcile"] = contract.get("reconcile", False)
    output = _mapping(contract.get("output"))
    if output.get("kind") == "json":
        result["output"] = {"kind": "json"}
    elif output.get("kind") == "schema" and isinstance(output.get("schema"), Mapping):
        result["output"] = {
            "kind": "schema",
            "schema": canonicalize_json_schema(
                cast(Mapping[str, JsonValue], output["schema"])
            ),
        }
    else:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return result


def _fields(
    value: Mapping[str, JsonValue],
    names: tuple[str, ...],
) -> "dict[str, JsonValue]":
    if any(name not in value for name in names):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return {name: value[name] for name in names}


def _mapping(value: object) -> "dict[str, JsonValue]":
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return dict(cast(Mapping[str, JsonValue], value))


__all__ = ["binding_identity_payload", "capability_identity_payload"]
