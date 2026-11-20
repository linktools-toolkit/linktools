#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physical owners for canonical runtime record addresses."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from ._content_reference_projection import CONTENT_REFERENCE_COLUMN_NAMES, CONTENT_REFERENCE_COLUMNS
from ._record_key import RECORD_KEY_MAX_BYTES, decode_key_components, encode_key_components

if TYPE_CHECKING:
    from sqlalchemy import MetaData
    from sqlalchemy.sql.selectable import FromClause
    from sqlalchemy.sql.elements import ColumnElement


EXECUTION_SCAN_STATES = frozenset((
    "PENDING_START", "STARTED", "RECOVERY_REQUIRED", "CANCELLING",
    "FINALIZING", "WAITING_DEFERRED",
))


def record_scan_kind(kind: str, state: str | None, *, prepared: bool = False) -> str:
    """Compute a query-only projection in the same write as its source record.

    Widen PAD SPACE aliases into the same candidate range on every backend;
    raw kind/state predicates still determine which rows a query returns.
    """
    if kind.rstrip(" ") == "execution":
        hot = state is not None and state.rstrip(" ") in EXECUTION_SCAN_STATES
        return "e:1" if hot or prepared else "e:0"
    return "k:" + kind


def record_scan_kinds(kind: str, states: frozenset[str] | None = None) -> tuple[str, ...]:
    """Select disjoint physical ranges without replacing the logical state filter."""
    if kind.rstrip(" ") != "execution":
        return ("k:" + kind,)
    if states is None:
        return ("e:1", "e:0")
    if not states:
        return ()
    # SQL collations can equate noncanonical strings (including trailing
    # spaces) with a hot state. Only exact known members prove a single range.
    if states <= EXECUTION_SCAN_STATES:
        return ("e:1",)
    return ("e:1", "e:0")


def record_owner_columns(table: "FromClause") -> tuple["ColumnElement", ...]:
    """Exclude physical scan projections from the common record wire shape."""
    return tuple(column for column in table.c if column.name != "scan_kind" and column.name not in CONTENT_REFERENCE_COLUMN_NAMES)


RUN_COUNTER_COLUMNS = ("event_high_water", "checkpoint_high_water", "interaction_high_water")
RECORD_OWNER_TABLES = ("ai_state_records", "ai_execution_records", "ai_step_tool_operations", "ai_step_run_admissions", "ai_task_graphs")
SCAN_KIND_OWNERS = frozenset(("ai_state_records", "ai_execution_records"))
_EXECUTION_ADDRESS_LENGTHS = {"execution": 6, "idempotency": 7}
_KIND_OWNERS = {"execution": "ai_execution_records", "idempotency": "ai_execution_records", "task_graph": "ai_task_graphs", "tool_operation": "ai_step_tool_operations", "agent_run": "ai_step_run_admissions"}
_CONSTANTS = {
    "ai_task_graphs": {"kind": "task_graph"},
    "ai_step_tool_operations": {"kind": "tool_operation", "payload_bytes": None},
    "ai_step_run_admissions": {"kind": "agent_run", "subject_id": None, "state": None,
        "lease_owner": None, "lease_fence": 0, "lease_expires_at": None, "payload_bytes": None},
}
_EDGE_OWNERS = {"ai_task_graphs": "task_graph", "ai_state_records": "record", "ai_step_tool_operations": "tool_record",
                "ai_step_run_admissions": "run_admission", "ai_execution_records": "record"}


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
        if parts[4] in _EXECUTION_ADDRESS_LENGTHS and (
                parts[3] != "execution" or len(parts) != _EXECUTION_ADDRESS_LENGTHS[parts[4]]):
            return "ai_state_records"
        return _KIND_OWNERS.get(parts[4], "ai_state_records")
    return "ai_state_records"


def record_owner_edge(key: bytes) -> str:
    """Retain the logical Native owner of the exact record address."""
    return _EDGE_OWNERS[record_owner_table(key)]


def record_owner_names(*, keys: Sequence[bytes] | None = None, kind: str | None = None) -> tuple[str, ...]:
    if keys is not None:
        return tuple(dict.fromkeys(record_owner_table(key) for key in keys))
    if kind == "archive_content":
        return ("ai_state_records",)
    if kind is not None:
        owner = _KIND_OWNERS.get(kind.rstrip(" "))
        return ("ai_state_records", owner) if owner is not None else ("ai_state_records",)
    return RECORD_OWNER_TABLES


def record_owner_rows(metadata: "MetaData", *, keys: Sequence[bytes] | None = None,
                      kind: str | None = None, owner: str | None = None,
                      include_run_counters: bool = False, scan_kind: str | None = None,
                      content_columns: Sequence[str] = ()) -> "FromClause":
    """Read a bounded owner set as one SQL statement, preserving global ordering."""
    from sqlalchemy import BigInteger, String, cast, literal, null, select, union_all

    if owner is not None and owner not in RECORD_OWNER_TABLES:
        raise ValueError("unknown record owner")
    names = (owner,) if owner is not None else record_owner_names(keys=keys, kind=kind)
    canonical = metadata.tables["ai_state_records"]
    if (len(names) == 1 and names[0] in SCAN_KIND_OWNERS and keys is not None and kind is None
            and scan_kind is None and not include_run_counters):
        table = metadata.tables[names[0]]
        if all(column in table.c for column in content_columns):
            return table
    selections = []
    for name in names:
        table = metadata.tables[name]
        constants = _CONSTANTS.get(name, {})
        # SQL type metadata alone does not type an inlined NULL on the server.
        # PostgreSQL resolves an untyped NULL in a derived owner to text before
        # outer UNIONs can see the corresponding canonical bytea/date column.
        # NULL has no text comparison semantics. Use a collation-free string
        # cast target: MySQL forbids COLLATE inside CAST(... AS CHAR ...).
        # Keep binary/date targets explicit for PostgreSQL derived UNIONs.
        selection = select(*(
            (cast(null(), String(column.type.length) if isinstance(column.type, String) else column.type) if constants[column.name] is None else
             literal(constants[column.name], type_=column.type, literal_execute=True)).label(column.name)
            if column.name in constants else table.c[column.name]
            for column in record_owner_columns(canonical)), *(
                (table.c[name] if name in table.c else cast(null(), BigInteger)).label(name)
                for name in RUN_COUNTER_COLUMNS if include_run_counters
            ), *(
                (table.c[column] if column in table.c else cast(null(), String(64)).label(column))
                for column in content_columns
            ))
        if name in SCAN_KIND_OWNERS and scan_kind is not None:
            selection = selection.where(table.c.scan_kind == literal(scan_kind, literal_execute=True))
        elif name in SCAN_KIND_OWNERS and kind is not None and kind.rstrip(" ") != "execution":
            selection = selection.where(table.c.scan_kind == literal("k:" + kind, literal_execute=True))
        selections.append(selection)
    return (selections[0] if len(selections) == 1 else union_all(*selections)).subquery("record_owners")


def record_owner_values(name: str, values: Mapping[str, object]) -> dict[str, object]:
    """Validate omitted invariants before writing a typed Step owner."""
    for key, value in _CONSTANTS.get(name, {}).items():
        if key in values and values[key] != value:
            raise ValueError(f"{name} requires {key}={value!r}")
    return {key: value for key, value in values.items()
            if key not in _CONSTANTS.get(name, {}) and (key != "scan_kind" or name in SCAN_KIND_OWNERS)
            and (key not in CONTENT_REFERENCE_COLUMN_NAMES or key in CONTENT_REFERENCE_COLUMNS.get(name, ()))}



__all__ = ["record_scan_kind", "EXECUTION_SCAN_STATES", "record_scan_kinds", "record_owner_columns", "RUN_COUNTER_COLUMNS", "RECORD_OWNER_TABLES", "SCAN_KIND_OWNERS", "record_owner_table", "record_owner_edge", "record_owner_names", "record_owner_rows", "record_owner_values"]

