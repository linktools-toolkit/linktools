#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recovery pages merge physical owners without scanning terminal graph history."""

from dataclasses import replace
from hashlib import sha256

import pytest
from sqlalchemy import String, create_engine, insert, text
from sqlalchemy.dialects import mysql, postgresql, sqlite

from linktools.ai.persistence._sql import _record_query_statement
from linktools.ai.persistence.schema import PersistenceDomain, build_runtime_sql_metadata
from linktools.ai.persistence.state import RecordQuery, record_key


def _row(index: int, *, state: str, opaque: bool = False, store: bytes = b"partition") -> dict:
    key = f"opaque-{index:05}".encode() if opaque else record_key(
        "recovery", "tenant", "task", "task_graph", (f"graph-{index:05}",),
    )
    scope = b"scope" if index % 2 else b"other-scope"
    parent = b"parent" if index % 3 else b"other-parent"
    order = b"x" * 128 + f"{index // 2:05}".encode()
    return dict(store_key=store, key=key, key_locator=sha256(key).hexdigest(),
                scope_key=scope, scope_locator=sha256(scope).hexdigest(),
                parent_key=parent, parent_locator=sha256(parent).hexdigest(),
                sort_key=order, sort_prefix=order[:128], state=state,
                storage_revision=0, payload_json={"index": index}, lease_fence=0)


@pytest.mark.parametrize("states", [None, frozenset(), frozenset({"PENDING"}),
    frozenset({"PENDING", "RUNNING"}), frozenset({"PENDING", "RUNNING", "SUCCEEDED"})])
@pytest.mark.parametrize("scope,parent,prefix", [(None, None, None), (b"scope", None, None),
    (None, b"parent", None), (b"scope", b"parent", b"x" * 128 + b"000")])
def test_graph_pages_preserve_scopes_and_raw_tie_cursors(
    states: frozenset[str] | None, scope: bytes | None, parent: bytes | None, prefix: bytes | None,
) -> None:
    metadata = build_runtime_sql_metadata(frozenset({PersistenceDomain.TASK}))
    graph = metadata.tables["ai_task_graphs"]
    generic = metadata.tables["ai_state_records"]
    engine = create_engine("sqlite://")
    metadata.create_all(engine)
    rows = [_row(i, state=("PENDING", "RUNNING", "SUCCEEDED")[i % 3], opaque=i % 7 == 0,
                 store=b"other" if i % 13 == 0 else b"partition") for i in range(100)]
    with engine.begin() as connection:
        for row in rows:
            opaque = row["key"].startswith(b"opaque")
            connection.execute(insert(generic if opaque else graph),
                               dict(row, kind="task_graph", scan_kind="k:task_graph") if opaque else row)
        # A different kind in the fallback table cannot enter the graph page.
        connection.execute(insert(generic), dict(_row(200, state="PENDING", opaque=True), kind="other", scan_kind="k:other"))
        expected = sorted((row for row in rows if row["store_key"] == b"partition"
            and (states is None or row["state"] in states)
            and (scope is None or row["scope_key"] == scope)
            and (parent is None or row["parent_key"] == parent)
            and (prefix is None or row["sort_key"].startswith(prefix))),
            key=lambda row: (row["sort_prefix"], row["sort_key"], row["key"]))
        query = RecordQuery(kind="task_graph", states=states, scope_key=scope,
                            parent_key=parent, sort_key_prefix=prefix, limit=2)
        actual = []
        while True:
            page = connection.execute(_record_query_statement(metadata, b"partition", query)).mappings().all()
            if not page:
                break
            actual.extend(row["key"] for row in page)
            query = replace(query, after_sort_key=page[-1]["sort_key"], after_key=page[-1]["key"])
        assert actual == [row["key"] for row in expected]
        for dialect in (sqlite.dialect(), mysql.dialect(), postgresql.dialect()):
            assert str(_record_query_statement(metadata, b"partition", query).compile(dialect=dialect))
    engine.dispose()



def test_graph_page_does_not_split_overlapping_state_ranges() -> None:
    metadata = build_runtime_sql_metadata(frozenset({PersistenceDomain.TASK}))
    graph = metadata.tables["ai_task_graphs"]
    # Reproduce the equality of MySQL's PAD SPACE state collation on SQLite.
    for name in ("ai_task_graphs", "ai_state_records"):
        metadata.tables[name].c.state.type = String(64, collation="PAD_SPACE")

    def compare(left: str, right: str) -> int:
        left, right = left.rstrip(" "), right.rstrip(" ")
        return (left > right) - (left < right)

    engine = create_engine("sqlite://")
    try:
        with engine.begin() as connection:
            connection.connection.driver_connection.create_collation("PAD_SPACE", compare)
            metadata.create_all(connection)
            rows = [_row(0, state="PENDING"), _row(1, state="PENDING ")]
            connection.execute(insert(graph), rows)
            query = RecordQuery(kind="task_graph", states=frozenset({"PENDING", "PENDING "}), limit=2)
            page = connection.execute(_record_query_statement(metadata, b"partition", query)).mappings().all()
            assert [row["key"] for row in page] == [row["key"] for row in rows]
    finally:
        engine.dispose()


@pytest.mark.parametrize("analyze", [False, True])
@pytest.mark.parametrize("terminal", [False, True])
def test_graph_recovery_scan_is_bounded_by_state_page(analyze: bool, terminal: bool) -> None:
    metadata = build_runtime_sql_metadata(frozenset({PersistenceDomain.TASK}))
    table = metadata.tables["ai_task_graphs"]
    engine = create_engine("sqlite://")
    metadata.create_all(engine)
    def make_row(index: int, state: str) -> dict:
        row = _row(index, state=state)
        if not terminal:
            # Distinct indexed prefixes isolate the page-bound behavior from
            # the documented cost of sorting a shared 128-byte prefix group.
            order = f"{index:06}".encode()
            row.update(sort_key=order, sort_prefix=order)
        return row
    with engine.begin() as connection:
        connection.execute(insert(table), [make_row(i, ("PENDING", "RUNNING")[i % 2]) for i in range(6)])
        query = RecordQuery(kind="task_graph", states=frozenset({"PENDING", "RUNNING"}), limit=129)
        statement = _record_query_statement(metadata, b"partition", query)
        work = []
        previous = 6
        for size in (1006, 10006):
            connection.execute(insert(table), [make_row(i, "SUCCEEDED" if terminal else ("PENDING", "RUNNING")[i % 2]) for i in range(previous, size)])
            previous = size
            if analyze:
                connection.execute(text("ANALYZE"))
            raw = connection.connection.driver_connection
            count = [0]
            def progress() -> int:
                count[0] += 1
                return 0
            raw.set_progress_handler(progress, 1)
            try:
                page = connection.execute(statement).mappings().all()
            finally:
                raw.set_progress_handler(None, 0)
            assert {row["key"] for row in page} == {_row(i, state="PENDING")["key"] for i in range(6 if terminal else 129)}
            work.append(count[0])
        # Neither terminal history nor later distinct-prefix active candidates
        # should expand an already-bounded page. VM work is not physical I/O.
        assert work[1] < work[0] * 2
    engine.dispose()

