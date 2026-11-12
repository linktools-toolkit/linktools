#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Indexed source/content joins for explicitly declared immutable references."""

import hashlib
from collections.abc import Sequence, Mapping
from typing import TYPE_CHECKING

from ._record_owner import record_owner_rows
from ._archive import archive_scope_parts
from ._step_payload_owner import payload_store_key, payload_store_locator

if TYPE_CHECKING:
    from sqlalchemy import MetaData
    from sqlalchemy.sql import Select


def content_reference_batch_size(path_count: int) -> int:
    """Bound the largest supported positional-bind expansion, including MySQL 5.7."""
    if type(path_count) is not int or not 1 <= path_count <= 16:
        raise ValueError("content reference paths must contain between one and sixteen slots")
    # Derived owner/refs occur in three UNION arms: two identity binds per
    # owner in each. Each path's JSON/CASE expressions repeat at most 32 binds;
    # the remaining tagged columns, scope predicates and link guards fit in 80.
    return min(200, (999 - 80 - 32 * path_count) // 6)


def content_reference_statement(
    metadata: "MetaData", *, dialect: str, owner_store_key: bytes, owner_keys: Sequence[bytes],
    content_store_key: bytes, content_scope_key: bytes, content_paths: Sequence[tuple[str, ...]],
    typed_payloads: bool, additional_records: Mapping[bytes, Sequence[bytes]] | None = None,
    owner_statement: "Select | None" = None,
) -> "Select":
    from sqlalchemy import JSON, LargeBinary, and_, case, cast, func, literal, select, true, type_coerce, union_all
    from sqlalchemy.dialects.mysql import BINARY

    owner_query = owner_statement
    if owner_query is None:
        table = record_owner_rows(metadata, keys=owner_keys)
        owner_query = select(table).where(
            table.c.store_key == owner_store_key,
            table.c.key_locator.in_(tuple(hashlib.sha256(key).hexdigest() for key in owner_keys)),
            table.c.key.in_(tuple(owner_keys)),
        )
    # MySQL 5.7 retains the same indexed plan through derived tables.
    owners = (owner_query.subquery("content_owners") if dialect == "mysql" else
              owner_query.cte("content_owners"))
    encoded = type_coerce(owners.c.payload_json, JSON)[()].as_string()
    payload = cast(encoded, JSON) if dialect == "postgresql" else type_coerce(encoded, JSON)
    slots = [select(literal(index).label("slot")) for index in range(len(content_paths))]
    slots = (slots[0] if len(slots) == 1 else union_all(*slots)).subquery("content_slots")
    identity = case(*((slots.c.slot == index, payload[path].as_string())
                      for index, path in enumerate(content_paths)))
    selected = select(identity.label("identity")).select_from(owners.join(slots, true())).subquery("declared_content")
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
        content.c.sort_prefix == func.substr(binary_identity, 1, 128), content.c.sort_key == binary_identity,
    ))
    native = metadata.tables.get("ai_step_messages") if typed_payloads else None
    native_nulls = (() if native is None else tuple(
        literal(None, type_=column.type).label("typed_" + column.name) for column in native.c))
    selections = [select(literal(0).label("source_kind"), *owners.c, *native_nulls),
                  select(literal(1).label("source_kind"), *content.c, *native_nulls).select_from(joined)]
    if native is not None:
        selections[-1] = selections[-1].where(func.substr(identity, 1, 18) != "execution-payload:")
        store = payload_store_key(*archive_scope_parts(content_scope_key))
        joined_native = references.join(native, and_(
            native.c.store_locator == payload_store_locator(store), native.c.store_key == store,
            native.c.sha256 == func.substr(identity, 19, 64),
            func.substr(identity, 1, 18) == "execution-payload:",
        ))
        selections.append(select(literal(2).label("source_kind"),
            *(literal(None, type_=column.type).label(column.name) for column in owners.c),
            *(column.label("typed_" + column.name) for column in native.c)).select_from(joined_native))
    for store_key, keys in (additional_records or {}).items():
        extra = record_owner_rows(metadata, keys=keys)
        selections.append(select(literal(3).label("source_kind"), *extra.c, *native_nulls).where(
            extra.c.store_key == store_key,
            extra.c.key_locator.in_(tuple(hashlib.sha256(key).hexdigest() for key in keys)),
            extra.c.key.in_(tuple(keys))))
    statement = select(union_all(*selections).subquery("owner_content_rows"))
    if dialect != "mysql":
        # Keep CTE materialization within this SELECT so callers can compose the
        # complete read plan without hoisting its private source names.
        statement = statement.add_cte(owners, references, nest_here=True)
        return select(statement.subquery("captured_content"))
    return statement


__all__ = ["content_reference_batch_size", "content_reference_statement"]
