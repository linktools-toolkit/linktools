#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Indexed source/content joins for explicitly declared immutable references."""

import hashlib
from collections.abc import Sequence, Mapping
from typing import TYPE_CHECKING

from ._content_reference_projection import EXECUTION_REFERENCE_COLUMNS, content_reference_slot
from ._record_owner import record_owner_rows, record_owner_columns, record_owner_table
from ._archive import archive_scope_parts
from ._step_payload_owner import payload_store_key, payload_store_locator

if TYPE_CHECKING:
    from sqlalchemy import MetaData
    from sqlalchemy.sql import Select


def content_reference_batch_size(path_count: int, *, native_path_count: int = 0) -> int:
    """Bound the largest supported positional-bind expansion, including MySQL 5.7."""
    if type(path_count) is not int or not 1 <= path_count <= 16:
        raise ValueError("content reference paths must contain between one and sixteen slots")
    # Derived owner/refs occur in three UNION arms: two identity binds per
    # owner in each. Each path's slot expressions repeat at most 32 binds;
    # the remaining tagged columns, scope predicates and link guards fit in 80.
    if type(native_path_count) is not int or not 0 <= native_path_count <= 4:
        raise ValueError("native paths must contain between zero and four slots")
    # The optional fixed-source arm repeats the owner selection and root slots
    # in MySQL derived SQL; budget those binds rather than assuming CTE reuse.
    return max(1, min(200, (999 - 80 - 32 * path_count - 32 * native_path_count)
                       // (8 if native_path_count else 6)))


def content_reference_statement(
    metadata: "MetaData", *, dialect: str, owner_store_key: bytes, owner_keys: Sequence[bytes],
    content_store_key: bytes, content_scope_key: bytes, content_paths: Sequence[tuple[str, ...]],
    typed_payloads: bool, additional_records: Mapping[bytes, Sequence[bytes]] | None = None,
    owner_statement: "Select | None" = None,
    native_root_paths: Sequence[tuple[str, ...]] = (),
    fixed_projection: bool = False,
) -> "Select":
    from sqlalchemy import JSON, Integer, LargeBinary, String, and_, case, cast, func, literal, null, select, true, type_coerce, union_all
    from sqlalchemy.dialects.mysql import BINARY

    content = tuple(content_reference_slot(path) for path in content_paths) if fixed_projection else ()
    roots = tuple(content_reference_slot(path) for path in native_root_paths) if fixed_projection else ()
    if (any(slot is None or slot.codec is None for slot in content)
            or any(slot is None or slot.codec is not None for slot in roots)):
        raise ValueError("SQL content capture requires declared lookup slots")
    owner_query = owner_statement
    if owner_query is None:
        columns = tuple(dict.fromkeys(slot.column for slot in (*content, *roots)))
        table = record_owner_rows(metadata, keys=owner_keys, content_columns=columns)
        owner_query = select(*record_owner_columns(table), *(table.c[name] for name in columns
                                     if name not in EXECUTION_REFERENCE_COLUMNS)).where(
            table.c.store_key == owner_store_key,
            table.c.key_locator.in_(tuple(hashlib.sha256(key).hexdigest() for key in owner_keys)),
            table.c.key.in_(tuple(owner_keys)),
        )
    # MySQL 5.7 retains the same indexed plan through derived tables.
    owners = (owner_query.subquery("content_owners") if dialect == "mysql" else
              owner_query.cte("content_owners"))
    pad_execution = any(name in owners.c for name in EXECUTION_REFERENCE_COLUMNS) or bool(
        additional_records and any(record_owner_table(key) == "ai_execution_records"
                                   for keys in additional_records.values() for key in keys))
    owner_columns = record_owner_columns(owners, pad_execution=pad_execution)
    reference_owners = (
        owner_query.with_only_columns(*(owner_query.selected_columns[slot.column] for slot in content),
                                      maintain_column_froms=True).subquery("content_reference_owners")
        if fixed_projection and dialect == "mysql" else owners
    )
    slots = [select(literal(index).label("slot")) for index in range(len(content_paths))]
    slots = (slots[0] if len(slots) == 1 else union_all(*slots)).subquery("content_slots")
    if fixed_projection:
        identities = tuple(literal(slot.codec + ":", literal_execute=True) + reference_owners.c[slot.column]
                           for slot in content)
    else:
        # Custom owners have no fixed-slot write contract. Preserve the
        # snapshot guarantee rather than loading their bodies in another query.
        encoded = type_coerce(owners.c.payload_json, JSON)[()].as_string()
        payload = cast(encoded, JSON) if dialect == "postgresql" else type_coerce(encoded, JSON)
        identities = tuple(func.coalesce(
            literal(slot.codec + ":", literal_execute=True) + owners.c[slot.column], payload[path].as_string())
            if (slot := content_reference_slot(path)) is not None and slot.codec is not None
            and slot.column in EXECUTION_REFERENCE_COLUMNS and slot.column in owners.c
            else payload[path].as_string() for path in content_paths)
    identity = case(*((slots.c.slot == index, value) for index, value in enumerate(identities)))
    selected = select(identity.label("identity")).select_from(reference_owners.join(slots, true())).subquery("declared_content")
    reference_query = select(selected.c.identity).where(selected.c.identity.is_not(None)).distinct()
    references = (reference_query.subquery("content_references") if dialect == "mysql" else
                  reference_query.cte("content_references"))
    identity = references.c.identity
    binary_identity = (func.convert_to(identity, "UTF8", type_=LargeBinary()) if dialect == "postgresql" else
                       cast(identity, BINARY() if dialect == "mysql" else LargeBinary()))
    content = metadata.tables["ai_state_records"].alias("content_source")
    joined = references.join(content, and_(
        content.c.store_key == content_store_key,
        content.c.scope_locator == hashlib.sha256(content_scope_key).hexdigest(),
        content.c.scope_key == content_scope_key, content.c.kind == "archive_content",
        content.c.scan_kind == literal("k:archive_content", literal_execute=True),
        content.c.sort_prefix == func.substr(binary_identity, 1, 128), content.c.sort_key == binary_identity,
    ))
    native = metadata.tables.get("ai_step_messages") if typed_payloads else None
    native_nulls = (() if native is None else tuple(
        literal(None, type_=column.type).label("typed_" + column.name) for column in native.c))
    selections = [select(literal(0).label("source_kind"), *owner_columns, *native_nulls),
                  select(literal(1).label("source_kind"), *record_owner_columns(content, pad_execution=pad_execution), *native_nulls).select_from(joined)]
    if native is not None:
        selections[-1] = selections[-1].where(func.substr(identity, 1, 18) != "execution-payload:")
        store = payload_store_key(*archive_scope_parts(content_scope_key))
        joined_native = references.join(native, and_(
            native.c.store_locator == payload_store_locator(store), native.c.store_key == store,
            native.c.sha256 == func.substr(identity, 19, 64),
            func.substr(identity, 1, 18) == "execution-payload:",
        ))
        selections.append(select(literal(2).label("source_kind"),
            *(literal(None, type_=column.type).label(column.name) for column in owner_columns),
            *(column.label("typed_" + column.name) for column in native.c)).select_from(joined_native))
    for store_key, keys in (additional_records or {}).items():
        extra = record_owner_rows(metadata, keys=keys)
        selections.append(select(literal(3).label("source_kind"), *record_owner_columns(extra, pad_execution=pad_execution), *native_nulls).where(
            extra.c.store_key == store_key,
            extra.c.key_locator.in_(tuple(hashlib.sha256(key).hexdigest() for key in keys)),
            extra.c.key.in_(tuple(keys))))
    if native_root_paths and "ai_step_sequences" in metadata.tables:
        from ..steps import native_store_key, native_store_locator

        nodes = metadata.tables["ai_step_sequences"].alias("fixed_native_nodes")
        messages = metadata.tables["ai_step_messages"].alias("fixed_native_messages")
        source_columns = tuple(nodes.c[name] for name in (
            "sha256", "message_sha256", "tree_count", "left_sha256", "right_sha256", "next_sha256")) + tuple(
            messages.c[name] for name in ("message", "compression", "raw_size", "payload_json"))
        null_columns = (cast(null(), Integer).label("native_slot"), *(cast(null(),
            String(column.type.length) if isinstance(column.type, String) else column.type
        ).label("native_" + column.name) for column in source_columns))
        selections = [selection.add_columns(*null_columns) for selection in selections]
        native_slots = [select(literal(index).label("slot")) for index in range(len(native_root_paths))]
        native_slots = (native_slots[0] if len(native_slots) == 1 else union_all(*native_slots)).subquery("native_slots")
        native_owners = (
            owner_query.with_only_columns(
                owner_query.selected_columns["key"], owner_query.selected_columns["store_key"],
                *(owner_query.selected_columns[slot.column] for slot in roots),
                maintain_column_froms=True,
            ).subquery("content_native_owners") if fixed_projection and dialect == "mysql" else owners
        )
        first_digests = (tuple(native_owners.c[slot.column] for slot in roots) if fixed_projection else
                         tuple(func.coalesce(owners.c[slot.column], payload[(*path, "slices", 0, "first")].as_string())
                               if (slot := content_reference_slot(path)) is not None and slot.column in owners.c
                               and slot.column in EXECUTION_REFERENCE_COLUMNS
                               else payload[(*path, "slices", 0, "first")].as_string() for path in native_root_paths))
        first = case(*((native_slots.c.slot == index, value) for index, value in enumerate(first_digests)))
        partition = native_store_key(*archive_scope_parts(content_scope_key))
        locator = native_store_locator(partition)
        native_join = native_owners.join(native_slots, true()).outerjoin(nodes, and_(
            nodes.c.store_key == partition, nodes.c.store_locator == locator, nodes.c.sha256 == first,
        )).outerjoin(messages, and_(
            messages.c.store_key == partition, messages.c.store_locator == locator,
            messages.c.sha256 == nodes.c.message_sha256,
        ))
        owner_locator = tuple(native_owners.c[column.name] if column.name in {"key", "store_key"} else cast(null(),
            String(column.type.length) if isinstance(column.type, String) else column.type).label(column.name)
            for column in owner_columns)
        selections.append(select(literal(4).label("source_kind"), *owner_locator, *native_nulls,
            native_slots.c.slot.label("native_slot"),
            *(column.label("native_" + column.name) for column in source_columns)
        ).select_from(native_join).where(first.is_not(None)))
    statement = select(union_all(*selections).subquery("owner_content_rows"))
    if dialect != "mysql":
        # Keep CTE materialization within this SELECT so callers can compose the
        # complete read plan without hoisting its private source names.
        statement = statement.add_cte(owners, references, nest_here=True)
        return select(statement.subquery("captured_content"))
    return statement


__all__ = ["content_reference_batch_size", "content_reference_statement"]

