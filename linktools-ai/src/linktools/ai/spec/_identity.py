#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal semantic projections for capability and Agent binding identities."""

import math
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

_TOOL_SEMANTIC_METADATA_FIELDS = (
    "linktools.ai.effect",
    "linktools.ai.plan_safe",
    "linktools.ai.tool_class",
    "linktools.ai.path_fields",
    "linktools.ai.compaction_keep_result",
    "linktools.ai.context_dedupe",
)


def agent_spec_identity_payload(
    contract: Mapping[str, JsonValue],
) -> "dict[str, JsonValue]":
    """Return the single semantic projection for an Agent declaration."""
    if not isinstance(contract, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    version = contract.get("version")
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != 1
    ):
        if isinstance(version, int) and not isinstance(version, bool):
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return _agent_spec_semantic(contract)


def bound_agent_spec_identity_payload(
    contract: Mapping[str, JsonValue],
) -> "dict[str, JsonValue]":
    """Return Agent execution semantics after the model route is resolved."""
    semantic = agent_spec_identity_payload(contract)
    semantic.pop("model")
    return semantic


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
        "agent_spec": bound_agent_spec_identity_payload(agent_spec),
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
        return agent_spec_identity_payload(contract)
    if kind == "tool":
        return _tool_semantic(contract)
    if kind == "skill":
        return _skill_semantic(contract)
    if kind == "mcp":
        return _mcp_semantic(contract)
    if kind == "capability":
        return _fields(contract, ("version", "revision", "defer_loading", "config"))
    if kind == "task":
        return _task_semantic(contract)
    return _fields(contract, ("version", "expander_id", "expander_version"))


def _mcp_semantic(contract: Mapping[str, JsonValue]) -> "dict[str, JsonValue]":
    result = _fields(contract, ("version", "id", "command"))
    execution_policy = contract.get("execution_policy")
    if execution_policy is not None:
        result["execution_policy"] = _execution_policy_semantic(
            execution_policy
        )
    args = contract.get("args")
    if not isinstance(args, list) or any(not isinstance(item, str) for item in args):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    result["args"] = args

    resource_root = contract.get("resource_root")
    has_resource_versions = "resource_versions" in contract
    resource_versions = contract.get("resource_versions")
    if has_resource_versions and not isinstance(resource_versions, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    if resource_root is None:
        if has_resource_versions:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (
            contract.get("resource_source_id") is not None
            or contract.get("resource_semantic_digest") is not None
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return result
    if not isinstance(resource_root, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if (
        not isinstance(resource_root.get("kind"), str)
        or not isinstance(resource_root.get("id"), str)
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    result["resource_root"] = {
        "kind": resource_root["kind"],
        "id": resource_root["id"],
    }
    if not has_resource_versions:
        if (
            contract.get("resource_source_id") is not None
            or contract.get("resource_semantic_digest") is not None
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return result
    source_id = contract.get("resource_source_id")
    digest = contract.get("resource_semantic_digest")
    if not isinstance(source_id, str) or not source_id or not _is_digest(digest):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    result["resource_semantic_digest"] = digest
    return result

def _execution_policy_semantic(value: JsonValue) -> "dict[str, JsonValue]":
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    policy = dict(cast(Mapping[str, JsonValue], value))
    if policy == {"version": 1, "boundary": "host-stdio"}:
        return policy
    hidden_paths = policy.get("hidden_paths")
    if (
        set(policy)
        != {
            "version",
            "boundary",
            "workspace_access",
            "hidden_paths",
            "network",
        }
        or policy.get("version") != 1
        or isinstance(policy.get("version"), bool)
        or policy.get("boundary") != "workspace-stdio"
        or policy.get("workspace_access") not in {"read", "read_write", "none"}
        or policy.get("network") != "isolated"
        or not isinstance(hidden_paths, list)
        or any(not isinstance(path, str) or not path for path in hidden_paths)
        or hidden_paths != sorted(set(hidden_paths))
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return policy


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
    semantic_metadata = {
        key: metadata[key]
        for key in _TOOL_SEMANTIC_METADATA_FIELDS
        if key in metadata
    }
    raw_return_schema = contract["return_schema"]
    return_schema = (
        None
        if raw_return_schema is None
        else canonicalize_json_schema(_mapping(raw_return_schema))
    )
    result: dict[str, JsonValue] = {
        "version": 1,
        "description": contract["description"],
        "parameters": canonicalize_json_schema(
            _mapping(contract["parameters"])
        ),
        "return_schema": return_schema,
        "strict": contract["strict"],
        "metadata": semantic_metadata,
    }
    max_retries = contract.get("max_retries")
    if max_retries is not None:
        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or max_retries < 0
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        result["max_retries"] = max_retries
    sequential = contract.get("sequential", False)
    if not isinstance(sequential, bool):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if sequential:
        result["sequential"] = True
    tool_kind = contract.get("kind", "function")
    if tool_kind not in {"function", "unapproved"}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if tool_kind != "function":
        result["kind"] = tool_kind
    timeout = contract.get("timeout")
    if timeout is not None:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        result["timeout"] = float(timeout)
    defer_loading = contract.get("defer_loading", False)
    if not isinstance(defer_loading, bool):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if defer_loading:
        result["defer_loading"] = True
    include_return_schema = contract.get("include_return_schema")
    if include_return_schema is not None:
        if not isinstance(include_return_schema, bool):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        result["include_return_schema"] = include_return_schema
    if "semantic_revision" in contract:
        revision = contract["semantic_revision"]
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        result["semantic_revision"] = revision
    return result


def _skill_semantic(contract: Mapping[str, JsonValue]) -> "dict[str, JsonValue]":
    result = _fields(contract, ("version", "id", "content"))
    if "description" in contract:
        result["description"] = contract["description"]
    source = contract.get("source")
    if source is None:
        return result
    source_value = _mapping(source)
    source_projection = _fields(source_value, ("source_id", "root"))
    resource_versions = source_value.get("resource_versions")
    if resource_versions is not None:
        if not isinstance(resource_versions, list) or not isinstance(
            source_value.get("sandbox_materialize"), bool
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        digest = source_value.get("resource_semantic_digest")
        if not _is_digest(digest):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        source_projection["resource_semantic_digest"] = digest
    elif (
        source_value.get("resource_semantic_digest") is not None
        or source_value.get("sandbox_materialize") is not None
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
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


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping(value: object) -> "dict[str, JsonValue]":
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return dict(cast(Mapping[str, JsonValue], value))


__all__ = [
    "agent_spec_identity_payload",
    "bound_agent_spec_identity_payload",
    "binding_identity_payload",
    "capability_identity_payload",
]
