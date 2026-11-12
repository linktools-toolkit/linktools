#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physical owners for canonical runtime record addresses."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from ._record_key import RECORD_KEY_MAX_BYTES, decode_key_components, encode_key_components

if TYPE_CHECKING:
    from sqlalchemy import MetaData
    from sqlalchemy.sql.selectable import FromClause


RECORD_OWNER_TABLES = ("ai_state_records", "ai_step_tool_operations", "ai_step_run_admissions")
_KIND_OWNERS = {"tool_operation": "ai_step_tool_operations", "agent_run": "ai_step_run_admissions"}
_CONSTANTS = {
    "ai_step_tool_operations": {"kind": "tool_operation", "payload_bytes": None},
    "ai_step_run_admissions": {"kind": "agent_run", "subject_id": None, "state": None,
        "lease_owner": None, "lease_fence": 0, "lease_expires_at": None, "payload_bytes": None},
}
_EDGE_OWNERS = {"ai_state_records": "record", "ai_step_tool_operations": "tool_record",
                "ai_step_run_admissions": "run_admission"}


def record_owner_table(key: bytes, *, store_key: bytes | None = None) -> str:
    """Resolve canonical addresses without interpreting opaque custom keys."""
    try:
        parts = decode_key_components(key, RECORD_KEY_MAX_BYTES)
    except (ValueError, UnicodeDecodeError):
        return "ai_state_records"
    if len(parts) == 6 and parts[0] == "record" and parts[3:5] == ("archive", "archive_content"):
        if parts[5].rpartition(":")[0] == "execution-payload":
            return ("ai_step_messages" if store_key == encode_key_components(("state-group-archive",), 128)
                    else "ai_state_records")
    if len(parts) >= 6 and parts[0] == "record":
        return _KIND_OWNERS.get(parts[4], "ai_state_records")
    return "ai_state_records"


def record_owner_edge(key: bytes) -> str:
    """Keep surrogate row IDs from distinct owner tables in distinct edge domains."""
    return _EDGE_OWNERS[record_owner_table(key)]


def record_owner_names(*, keys: Sequence[bytes] | None = None, kind: str | None = None) -> tuple[str, ...]:
    if keys is not None:
        return tuple(dict.fromkeys(record_owner_table(key) for key in keys))
    if kind == "archive_content":
        return ("ai_state_records",)
    if kind is not None:
        owner = _KIND_OWNERS.get(kind)
        return ("ai_state_records", owner) if owner is not None else ("ai_state_records",)
    return RECORD_OWNER_TABLES


def record_owner_rows(metadata: "MetaData", *, keys: Sequence[bytes] | None = None,
                      kind: str | None = None, owner: str | None = None) -> "FromClause":
    """Read a bounded owner set as one SQL statement, preserving global ordering."""
    from sqlalchemy import literal, select, union_all

    if owner is not None and owner not in RECORD_OWNER_TABLES:
        raise ValueError("unknown record owner")
    names = (owner,) if owner is not None else record_owner_names(keys=keys, kind=kind)
    if names == ("ai_state_records",):
        return metadata.tables[names[0]]
    canonical = metadata.tables["ai_state_records"]
    selections = []
    for name in names:
        table = metadata.tables[name]
        constants = _CONSTANTS.get(name, {})
        selections.append(select(*(literal(constants[column.name], type_=column.type, literal_execute=True).label(column.name)
                                   if column.name in constants else table.c[column.name]
                                   for column in canonical.c)))
    return (selections[0] if len(selections) == 1 else union_all(*selections)).subquery("record_owners")


def record_owner_values(name: str, values: Mapping[str, object]) -> dict[str, object]:
    """Validate omitted invariants before writing a typed Step owner."""
    for key, value in _CONSTANTS.get(name, {}).items():
        if key in values and values[key] != value:
            raise ValueError(f"{name} requires {key}={value!r}")
    return {key: value for key, value in values.items() if key not in _CONSTANTS.get(name, {})}



__all__ = ["RECORD_OWNER_TABLES", "record_owner_table", "record_owner_edge", "record_owner_names", "record_owner_rows", "record_owner_values"]
