#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task capability capture manifest declaration contracts."""

import re
from collections.abc import Callable, Mapping

from ...core import ImmutableJsonMapping, JsonValue
from ...errors import AIError, ErrorCode
from ...spec import canonicalize_json_schema

TASK_CAPABILITY_CAPTURE_FORMAT_VERSION = 1
_TASK_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_BUILTIN_TASK_IDENTITIES = {
    ("linktools.ai.agent", 1),
    ("linktools.ai.input", 1),
}


def read_task_capability_capture_declarations(
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
    if not isinstance(value, Mapping) or set(value) != {
        "version",
        "id",
        "revision",
        "effect_policy",
        "output_contract",
        "reconcile",
    }:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    identity = _task_identity(value.get("id"), value.get("revision"))
    output_contract = value.get("output_contract")
    contract_version = value.get("version")
    effect_policy = value.get("effect_policy")
    if (
        isinstance(contract_version, bool)
        or not isinstance(contract_version, int)
        or contract_version != 1
        or not isinstance(effect_policy, str)
        or effect_policy not in {"none", "replay_safe", "non_replay_safe"}
        or not isinstance(value.get("reconcile"), bool)
        or not isinstance(output_contract, Mapping)
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
    if identity[0].startswith("linktools.ai.") and identity not in _BUILTIN_TASK_IDENTITIES:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
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
    "TASK_CAPABILITY_CAPTURE_FORMAT_VERSION",
    "read_task_capability_capture_declarations",
    "task_declaration_identity",
    "task_expander_declaration_identity",
]
