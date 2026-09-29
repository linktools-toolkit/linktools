#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task graph binding capture manifest declaration contracts."""

import re
from collections.abc import Callable, Mapping

from ...core import ImmutableJsonMapping, JsonValue
from ...errors import AIError, ErrorCode
from ...spec import canonicalize_json_schema

TASK_GRAPH_BINDING_CAPTURE_FORMAT_VERSION = 1
TASK_GRAPH_BINDING_CAPTURE_MANIFEST_KEYS = frozenset(
    {
        "kind",
        "format_version",
        "namespace",
        "tenant_id",
        "graph_id",
        "request_digest",
        "tasks",
        "expanders",
    }
)
_TASK_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_TASK_TYPE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")


def read_task_graph_binding_capture_declarations(
    tasks: object,
    expanders: object,
) -> tuple[
    dict[tuple[str, int], Mapping[str, JsonValue]],
    dict[tuple[str, int], Mapping[str, JsonValue]],
]:
    """Decode and validate captured Task and expander declarations."""
    return (
        _read_declarations(tasks, task_declaration_identity),
        _read_declarations(expanders, task_expander_declaration_identity),
    )


def task_declaration_identity(value: object) -> tuple[str, int]:
    if not isinstance(value, Mapping) or not {
        "version",
        "id",
        "revision",
        "type",
        "effect_policy",
        "output_contract",
        "reconcile",
    }.issubset(value) or set(value) - {
        "version",
        "id",
        "revision",
        "type",
        "effect_policy",
        "output_contract",
        "reconcile",
        "config",
    }:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    identity = _task_identity(value.get("id"), value.get("revision"))
    output_contract = value.get("output_contract")
    contract_version = value.get("version")
    task_type = value.get("type")
    effect_policy = value.get("effect_policy")
    config = value.get("config")
    if (
        isinstance(contract_version, bool)
        or not isinstance(contract_version, int)
        or contract_version != 1
        or not isinstance(task_type, str)
        or _TASK_TYPE.fullmatch(task_type) is None
        or not isinstance(effect_policy, str)
        or effect_policy not in {"none", "replay_safe", "non_replay_safe"}
        or not isinstance(value.get("reconcile"), bool)
        or not isinstance(output_contract, Mapping)
        or ("config" in value and not isinstance(config, Mapping))
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if output_contract.get("kind") == "json":
        if set(output_contract) != {"kind"}:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    elif output_contract.get("kind") == "schema":
        schema = output_contract.get("schema")
        if (
            set(output_contract) != {"kind", "schema"}
            or not isinstance(schema, Mapping)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            canonicalize_json_schema(schema)
        except AIError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    else:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if task_type == "agent":
        if not isinstance(config, Mapping) or set(config) != {
            "agent_id",
            "agent_revision",
            "binding_contract",
            "input_mode",
        }:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (
            not isinstance(config.get("agent_id"), str)
            or not config.get("agent_id")
            or isinstance(config.get("agent_revision"), bool)
            or not isinstance(config.get("agent_revision"), int)
            or config["agent_revision"] < 1
            or config.get("input_mode") not in {"literal", "projected"}
            or not isinstance(config.get("binding_contract"), Mapping)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        from ...agent import AgentBindingContract

        AgentBindingContract.from_payload(config["binding_contract"])
    return identity


def task_expander_declaration_identity(value: object) -> tuple[str, int]:
    if not isinstance(value, Mapping) or set(value) != {
        "version",
        "id",
        "revision",
    }:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    identity = _task_identity(value.get("id"), value.get("revision"))
    contract_version = value.get("version")
    if (
        isinstance(contract_version, bool)
        or not isinstance(contract_version, int)
        or contract_version != 1
        or identity[0].startswith("linktools.ai.")
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return identity


def _read_declarations(
    values: object,
    identity_of: Callable[[object], tuple[str, int]],
) -> dict[tuple[str, int], Mapping[str, JsonValue]]:
    if not isinstance(values, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    declarations: dict[tuple[str, int], Mapping[str, JsonValue]] = {}
    for value in values:
        if not isinstance(value, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            declaration = ImmutableJsonMapping(value)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        identity = identity_of(declaration)
        if identity in declarations:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        declarations[identity] = declaration
    return dict(sorted(declarations.items()))


def _task_identity(task_id: object, revision: object) -> tuple[str, int]:
    if (
        not isinstance(task_id, str)
        or _TASK_ID.fullmatch(task_id) is None
        or isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 1
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return task_id, revision


__all__ = [
    "TASK_GRAPH_BINDING_CAPTURE_FORMAT_VERSION",
    "TASK_GRAPH_BINDING_CAPTURE_MANIFEST_KEYS",
    "read_task_graph_binding_capture_declarations",
    "task_declaration_identity",
    "task_expander_declaration_identity",
]
