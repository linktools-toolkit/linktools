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
            rows = connection.execute(
                "SELECT kind, count(*), sum(length(payload_json)), "
                "sum(coalesce(length(payload_bytes), 0)) "
                "FROM ai_state_records GROUP BY kind"
            ).fetchall()
        print(json.dumps({
            "root": str(root),
            "kinds": rows,
            "json_plus_binary": sum((row[2] or 0) + (row[3] or 0) for row in rows),
            "file_bytes": path.stat().st_size,
            "sql": {"/".join(key): value for key, value in sorted(counts.items())},
        }))


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]).resolve()))
