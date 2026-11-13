#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic SQLite byte/SQL probe; pass the isolated checkout root to measure."""

import asyncio
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import random
import sqlite3
import string
import sys
import tempfile


def measure_table_bytes(connection: sqlite3.Connection) -> dict[str, dict[str, int]]:
    """Count every AI table's stored text/blob bytes, including referenced bodies.

    This is logical column data, not index pages, numeric fields or file size.
    Discover tables so baseline records and current Step/Object owners use the
    same accounting, without counting references as additional body copies.
    """
    result = {}
    tables = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name GLOB 'ai_*' ORDER BY name"
    ).fetchall()
    for (name,) in tables:
        quoted = '"' + name.replace('"', '""') + '"'
        columns = connection.execute(f"PRAGMA table_info({quoted})").fetchall()
        sizes = []
        for column in columns:
            key = '"' + column[1].replace('"', '""') + '"'
            sizes.append(f"CASE WHEN typeof({key}) IN ('text', 'blob') "
                         f"THEN length(CAST({key} AS BLOB)) ELSE 0 END")
        count, size = connection.execute(
            f"SELECT count(*), coalesce(sum({' + '.join(sizes)}), 0) FROM {quoted}"
        ).fetchone()
        result[name] = {"rows": count, "text_and_blob_bytes": size}
    return result


async def main(root: Path) -> None:
    # Select the measured checkout before importing project modules or fixtures.
    sys.path[:0] = [str(root), str(root / "linktools-ai/src"), str(root / "linktools/src")]
    from sqlalchemy import event
    from linktools.ai.persistence import Persistence, PersistenceStorage
    from linktools.ai.storage import StoredPayload
    from tests.ai.test_tool_execution_boundary import _request

    rng = random.Random(13)
    counts: Counter[tuple[str, str]] = Counter()
    phase = ""
    payloads = [
        StoredPayload.inline_bytes("".join(rng.choices(
            string.ascii_letters + string.digits + " ", k=size,
        )).encode())
        for size in [369816] + [5500] * 12
    ]

    def sql(_connection: object, _cursor: object, statement: str, *args: object) -> None:
        counts[(phase, statement.split()[0].upper())] += 1

    with tempfile.TemporaryDirectory(prefix="tool-content-") as temporary:
        path = Path(temporary) / "db.sqlite"
        async with Persistence.open(
            "tool-size", tenant_id="tenant", storage=PersistenceStorage.sqlite(path),
        ) as state:
            tools = state.checkpoint.tools
            await tools.state_store.storage_group.open_archive(create=True)
            engine = tools.state_store.storage_group.context.engine.sync_engine
            event.listen(engine, "before_cursor_execute", sql)
            try:
                for index, payload in enumerate(payloads):
                    phase = "admit"
                    operation = await tools.admit(replace(
                        _request(), tool_operation_id=f"op-{index}", tool_call_id=f"call-{index}",
                    ))
                    phase = "complete"
                    await tools.complete_payload(
                        operation.tool_operation_id, tenant_id="tenant", owner=operation.owner,
                        fence=operation.fence, result_payload=payload,
                    )
                phase = "batch_read"
                values = await tools.get_operations(
                    tuple(f"op-{index}" for index in range(13)), tenant_id="tenant",
                )
                assert tuple(value.result_payload for value in values) == tuple(payloads)
            finally:
                event.remove(engine, "before_cursor_execute", sql)
        with sqlite3.connect(path) as connection:
            table_bytes = measure_table_bytes(connection)
        print(json.dumps({
            "root": str(root),
            "tables": table_bytes,
            "text_and_blob_bytes": sum(row["text_and_blob_bytes"] for row in table_bytes.values()),
            "file_bytes": path.stat().st_size,
            "sql": {"/".join(key): value for key, value in sorted(counts.items())},
        }))


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]).resolve()))
