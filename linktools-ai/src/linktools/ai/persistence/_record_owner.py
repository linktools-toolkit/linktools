#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physical owners for canonical runtime record addresses."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from ..core import (
    AGENT_RUN_ID_MAX_BYTES, EXECUTION_ID_MAX_BYTES, EXECUTION_REQUEST_KEY_MAX_BYTES,
    PERSISTENCE_NAMESPACE_MAX_BYTES, RESOURCE_ID_MAX_BYTES, TASK_GRAPH_ID_MAX_BYTES,
    TENANT_ID_MAX_BYTES, TOOL_OPERATION_ID_MAX_BYTES, validate_lease_owner,
)
from ..errors import AIError, ErrorCode
from ._content_reference_projection import CONTENT_REFERENCE_COLUMN_NAMES, CONTENT_REFERENCE_COLUMNS, EXECUTION_REFERENCE_COLUMNS
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


EXECUTION_SHELL_KIND = "execution_shell"
EXECUTION_AUTHORITY_COLUMNS = frozenset({
    "turn_slot_kind", "turn_slot_key", "turn_token", "turn_fence", "turn_expiry", "turn_revision",
    "turn_request_route", "turn_occupied_route", "turn_receipt",
})


def record_owner_columns(table: "FromClause", *, pad_execution: bool = False) -> tuple["ColumnElement", ...]:
    """Return the common wire row and any canonical execution digest cells."""
    from sqlalchemy import String, cast, null

    base = tuple(column for column in table.c if column.name != "scan_kind"
                 and column.name not in CONTENT_REFERENCE_COLUMN_NAMES and column.name not in EXECUTION_AUTHORITY_COLUMNS)
    return base + tuple(
        table.c[name] if name in table.c else cast(null(), String(64)).label(name)
        for name in EXECUTION_REFERENCE_COLUMNS if pad_execution or name in table.c
    )


_SCOPE_OWNER_MAX_BYTES = PERSISTENCE_NAMESPACE_MAX_BYTES + TENANT_ID_MAX_BYTES

RUN_COUNTER_COLUMNS = ("event_high_water", "checkpoint_high_water", "interaction_high_water")
# Framed execution and idempotency writer addresses share this physical owner.
EXECUTION_RECORD_KEY_MAX_BYTES = _SCOPE_OWNER_MAX_BYTES + max(
    36 + EXECUTION_ID_MAX_BYTES,
    40 + RESOURCE_ID_MAX_BYTES + EXECUTION_REQUEST_KEY_MAX_BYTES,
)
# Execution, idempotency and authority-shell rows order by their framed key.
# Timestamp-prefixed sort addresses belong to other State owners.
EXECUTION_SORT_KEY_MAX_BYTES = EXECUTION_RECORD_KEY_MAX_BYTES
# Six framed components: record, namespace, tenant, task, task_graph, graph ID.
TASK_GRAPH_KEY_MAX_BYTES = 32 + _SCOPE_OWNER_MAX_BYTES + TASK_GRAPH_ID_MAX_BYTES
# Six framed components: record, namespace, tenant, checkpoint, tool_operation, ID.
TOOL_OPERATION_KEY_MAX_BYTES = 42 + _SCOPE_OWNER_MAX_BYTES + TOOL_OPERATION_ID_MAX_BYTES
# Six framed components: record, namespace, tenant, domain, agent_run, run ID.
_RUN_ADMISSION_DOMAINS = frozenset(("trace", "checkpoint", "conversation"))
RUN_ADMISSION_KEY_MAX_BYTES = 27 + max(map(len, _RUN_ADMISSION_DOMAINS)) + _SCOPE_OWNER_MAX_BYTES + AGENT_RUN_ID_MAX_BYTES
# Seven framed components: role, namespace, tenant, domain, agent_run, relation, ID.
RUN_ADMISSION_SCOPE_MAX_BYTES = 40 + max(map(len, _RUN_ADMISSION_DOMAINS)) + _SCOPE_OWNER_MAX_BYTES + RESOURCE_ID_MAX_BYTES
RUN_ADMISSION_PARENT_MAX_BYTES = 35 + max(map(len, _RUN_ADMISSION_DOMAINS)) + _SCOPE_OWNER_MAX_BYTES + AGENT_RUN_ID_MAX_BYTES
# Tool ledger accepts both exact resource IDs at the 128-byte public boundary.
TOOL_OPERATION_SCOPE_MAX_BYTES = 52 + _SCOPE_OWNER_MAX_BYTES + RESOURCE_ID_MAX_BYTES
TOOL_OPERATION_PARENT_MAX_BYTES = 53 + _SCOPE_OWNER_MAX_BYTES + RESOURCE_ID_MAX_BYTES
# Execution can carry a session scope or an EXECUTION idempotency resource scope.
EXECUTION_SCOPE_MAX_BYTES = _SCOPE_OWNER_MAX_BYTES + max(44 + RESOURCE_ID_MAX_BYTES, 58 + EXECUTION_ID_MAX_BYTES)
EXECUTION_PARENT_MAX_BYTES = 47 + _SCOPE_OWNER_MAX_BYTES + EXECUTION_ID_MAX_BYTES
RECORD_OWNER_RELATION_WIDTHS = {
    "ai_execution_records": (EXECUTION_SCOPE_MAX_BYTES, EXECUTION_PARENT_MAX_BYTES),
    "ai_step_run_admissions": (RUN_ADMISSION_SCOPE_MAX_BYTES, RUN_ADMISSION_PARENT_MAX_BYTES),
    "ai_step_tool_operations": (TOOL_OPERATION_SCOPE_MAX_BYTES, TOOL_OPERATION_PARENT_MAX_BYTES),
}
RECORD_OWNER_TABLES = ("ai_state_records", "ai_execution_records", "ai_step_tool_operations", "ai_step_run_admissions", "ai_task_graphs")
SCAN_KIND_OWNERS = frozenset(("ai_state_records", "ai_execution_records"))
_EXECUTION_ADDRESS_IDENTITIES = {
    "execution": (EXECUTION_ID_MAX_BYTES,),
    "idempotency": (RESOURCE_ID_MAX_BYTES, EXECUTION_REQUEST_KEY_MAX_BYTES),
}
_KIND_OWNERS = {"execution": "ai_execution_records", "idempotency": "ai_execution_records", "task_graph": "ai_task_graphs", "tool_operation": "ai_step_tool_operations", "agent_run": "ai_step_run_admissions"}
_CONSTANTS = {
    "ai_execution_records": {"payload_bytes": None},
    "ai_task_graphs": {"kind": "task_graph", "scope_key": None, "parent_key": None,
                       "scope_locator": None, "parent_locator": None},
    "ai_step_tool_operations": {"kind": "tool_operation", "payload_bytes": None},
    "ai_step_run_admissions": {"kind": "agent_run", "subject_id": None, "state": None,
        "lease_owner": None, "lease_fence": 0, "lease_expires_at": None, "payload_bytes": None},
}
_EDGE_OWNERS = {"ai_task_graphs": "task_graph", "ai_state_records": "record", "ai_step_tool_operations": "tool_record",
                "ai_step_run_admissions": "run_admission", "ai_execution_records": "record"}


def is_run_admission_address(parts: Sequence[str]) -> bool:
    """Recognize the bounded owner prefix and run ID shared by record and stream addresses."""
    return (len(parts) == 6 and parts[3] in _RUN_ADMISSION_DOMAINS
            and all(1 <= len(value.encode("utf-8")) <= width for value, width in
                    zip(parts[1:3], (PERSISTENCE_NAMESPACE_MAX_BYTES, TENANT_ID_MAX_BYTES)))
            and 1 <= len(parts[5].encode("utf-8")) <= AGENT_RUN_ID_MAX_BYTES)


def record_owner_table(key: bytes, *, store_key: bytes | None = None) -> str:
    """Resolve canonical addresses without interpreting opaque custom keys."""
    try:
        parts = decode_key_components(key, RECORD_KEY_MAX_BYTES)
    except (ValueError, UnicodeDecodeError):
        return "ai_state_records"
    if len(parts) == 6 and parts[0] == "record" and parts[3:5] == ("archive", "archive_content"):
        if parts[5].rpartition(":")[0] == "execution-payload":
            return ("ai_messages" if store_key == encode_key_components(("state-group-archive",), 128)
                    else "ai_state_records")
    if len(parts) >= 6 and parts[0] == "record":
        if parts[4] == "task_graph" and (
                len(parts) != 6 or parts[3] != "task" or len(key) > TASK_GRAPH_KEY_MAX_BYTES
                or len(parts[5].encode("utf-8")) > TASK_GRAPH_ID_MAX_BYTES
        ):
            return "ai_state_records"
        if parts[4] == "tool_operation" and (
                len(parts) != 6 or parts[3] != "checkpoint" or len(key) > TOOL_OPERATION_KEY_MAX_BYTES
                or len(parts[5].encode("utf-8")) > TOOL_OPERATION_ID_MAX_BYTES
        ):
            return "ai_state_records"
        if parts[4] == "agent_run" and not is_run_admission_address(parts):
            return "ai_state_records"
        execution_identity_widths = _EXECUTION_ADDRESS_IDENTITIES.get(parts[4])
        if execution_identity_widths is not None and (
                parts[3] != "execution" or len(parts) != 5 + len(execution_identity_widths)
                or any(not 1 <= len(value.encode("utf-8")) <= width for value, width in
                       zip((*parts[1:3], *parts[5:]), (PERSISTENCE_NAMESPACE_MAX_BYTES, TENANT_ID_MAX_BYTES, *execution_identity_widths)))
        ):
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
    if "ai_execution_records" in names:
        content_columns = tuple(dict.fromkeys((*EXECUTION_REFERENCE_COLUMNS, *content_columns)))
    if (len(names) == 1 and names[0] == "ai_state_records" and keys is not None and kind is None
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
        # Execution bucket pages already apply the requested kind outside
        # this branch. A second kind predicate can defeat the SQLite bucket index.
        execution_bucket = scan_kind is not None and kind is not None and kind.rstrip(" ") == "execution"
        if name == "ai_execution_records" and not execution_bucket:
            selection = selection.where(table.c.kind != literal(EXECUTION_SHELL_KIND, literal_execute=True))
        if name in SCAN_KIND_OWNERS and scan_kind is not None:
            selection = selection.where(table.c.scan_kind == literal(scan_kind, literal_execute=True))
        elif name in SCAN_KIND_OWNERS and kind is not None and kind.rstrip(" ") != "execution":
            selection = selection.where(table.c.scan_kind == literal("k:" + kind, literal_execute=True))
        selections.append(selection)
    return (selections[0] if len(selections) == 1 else union_all(*selections)).subquery("record_owners")


def record_owner_values(name: str, values: Mapping[str, object]) -> dict[str, object]:
    """Validate and shape values for the record's physical owner."""
    if name == "ai_execution_records":
        if "key" in values and "kind" in values:
            key = values["key"]
            if not isinstance(key, bytes) or values["kind"] != decode_key_components(key, RECORD_KEY_MAX_BYTES)[4]:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if "sort_key" in values and (
            type(values["sort_key"]) is not bytes or not 1 <= len(values["sort_key"]) <= EXECUTION_SORT_KEY_MAX_BYTES
        ):
            raise ValueError("execution sort key exceeds its bounded record ordering size")
    lease_owner = values.get("lease_owner")
    if lease_owner is not None:
        validate_lease_owner(lease_owner)
    for key, value in _CONSTANTS.get(name, {}).items():
        if key in values and values[key] != value:
            raise ValueError(f"{name} requires {key}={value!r}")
    return {key: value for key, value in values.items()
            if key not in _CONSTANTS.get(name, {}) and (key != "scan_kind" or name in SCAN_KIND_OWNERS)
            and (key not in CONTENT_REFERENCE_COLUMN_NAMES or key in CONTENT_REFERENCE_COLUMNS.get(name, ()))}



__all__ = ["EXECUTION_SHELL_KIND", "TASK_GRAPH_KEY_MAX_BYTES", "EXECUTION_AUTHORITY_COLUMNS", "record_scan_kind", "EXECUTION_SCAN_STATES", "record_scan_kinds", "record_owner_columns", "RUN_COUNTER_COLUMNS", "RUN_ADMISSION_KEY_MAX_BYTES", "is_run_admission_address", "RECORD_OWNER_TABLES", "SCAN_KIND_OWNERS", "record_owner_table", "record_owner_edge", "record_owner_names", "record_owner_rows", "record_owner_values"]

