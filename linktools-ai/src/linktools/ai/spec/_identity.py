#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable references for named behavior and digest inputs for Agent bindings."""

from collections.abc import Mapping
from typing import cast

from ..core import JsonValue, canonical_sha256
from ..errors import AIError, ErrorCode
from ._schema import canonicalize_json_schema

_CONTRIBUTION_KINDS = frozenset(
    {"agent", "tool", "skill", "mcp", "capability", "task", "task_expander"}
)


def agent_ref_payload(
    contract: Mapping[str, JsonValue],
) -> "dict[str, JsonValue]":
    _require_format_v1(contract)
    return _ref_payload("agent", _text(contract.get("id")), _revision(contract))


def capability_ref_payload(
    kind: str,
    identity: str,
    contract: Mapping[str, JsonValue],
) -> "dict[str, JsonValue]":
    if kind not in _CONTRIBUTION_KINDS or not isinstance(identity, str) or not identity:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if not isinstance(contract, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    _require_format_v1(contract)
    if kind in {"agent", "skill", "mcp"}:
        if contract.get("id") != identity:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return _ref_payload(kind, identity, _revision(contract))
    if kind in {"tool", "capability"}:
        return _ref_payload(kind, identity, _revision(contract))
    if kind == "task":
        ref_id = _text(contract.get("id"))
        revision = _positive_int(contract.get("revision"))
        if identity != ref_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return _ref_payload(kind, ref_id, revision)
    ref_id = _text(contract.get("id"))
    revision = _positive_int(contract.get("revision"))
    if identity != ref_id:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return _ref_payload(kind, ref_id, revision)


def binding_digest_payload(
    payload: Mapping[str, JsonValue],
) -> "dict[str, JsonValue]":
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
    _require_format_v1(payload)
    agent_spec = _mapping(payload["agent_spec"])
    base_model = _mapping(payload["base_model"])
    selected = payload["selected"]
    subagents = payload["subagents"]
    if not isinstance(selected, list) or not isinstance(subagents, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    selected_refs: list[JsonValue] = []
    for item in selected:
        value = _mapping(item)
        kind = value.get("kind")
        identity = value.get("id")
        contract = value.get("contract")
        if (
            not isinstance(kind, str)
            or not isinstance(identity, str)
            or not isinstance(contract, Mapping)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        selected_refs.append(
            capability_ref_payload(
                kind,
                identity,
                cast(Mapping[str, JsonValue], contract),
            )
        )

    subagent_refs: list[JsonValue] = []
    for item in subagents:
        value = _mapping(item)
        if value.get("kind") != "agent":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        subagent_refs.append(
            _ref_payload(
                "agent",
                _text(value.get("id")),
                _positive_int(value.get("revision", 1)),
            )
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
        "model": dict(base_model),
        "selected": selected_refs,
        "subagents": subagent_refs,
        "output": {
            "mode": output_mode,
            "schema": canonicalize_json_schema(
                cast(Mapping[str, JsonValue], output_schema)
            ),
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
        child_refs.sort(
            key=lambda value: cast(
                str,
                cast(Mapping[str, JsonValue], value["agent"])["id"],
            )
        )
        result["subagent_bindings"] = child_refs
    return result


def _ref_payload(kind: str, identity: str, revision: int) -> "dict[str, JsonValue]":
    return {
        "kind": kind,
        "id": identity,
        "revision": revision,
    }


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
    return dict(cast(Mapping[str, JsonValue], value))


__all__ = [
    "agent_ref_payload",
    "binding_digest_payload",
    "capability_ref_payload",
]
