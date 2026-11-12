#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone SQLite publication/read probe; does not run a test suite."""

import asyncio
import json
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path

from pydantic_ai.messages import ModelRequest, ToolReturnPart
from sqlalchemy import event

from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.core import canonical_json_bytes
from linktools.ai.persistence.codec import decode_envelope, native_tool_result_reference

from tests.ai.test_native_text_result_runtime import _prepare, _publish
from tests.ai.test_native_tool_result_runtime import _settle


async def main() -> None:
    config = PersistenceStorage.sqlite(Path(tempfile.mkdtemp(prefix="native-tool-measure-")) / "state.db")
    counts: Counter[str] = Counter()
    report: dict[str, object] = {}

    def sql(connection: object, cursor: object, statement: str, parameters: object,
            context: object, executemany: bool) -> None:
        counts[statement.split()[0]] += 1

    async with Persistence.open("native-result", tenant_id="tenant", storage=config) as state:
        commands, commit, current, checkpoint = await _prepare(state, "done", "done")
        body = "tool receipt " * 10000
        operation, payload = await _settle(state, body)
        checkpoint = replace(checkpoint, messages=[ModelRequest(parts=[ToolReturnPart("tool", body, tool_call_id="call")]),
                                                   *checkpoint.messages])
        engine = state.checkpoint.tools.state_store.storage_group.context.engine.sync_engine
        event.listen(engine, "before_cursor_execute", sql)
        try:
            await _publish(commands, commit, current, checkpoint)
            report["whole_terminal_publication"] = dict(counts)
            counts.clear()
            loaded = await state.checkpoint.tools.get_operation(operation.tool_operation_id, tenant_id="tenant")
            assert loaded.result_payload == payload
            report["warm_tool_read"] = dict(counts)
        finally:
            event.remove(engine, "before_cursor_execute", sql)
        records = await state.checkpoint.tools.state_store.read(lambda transaction: transaction.scan_records())
        record = next(record for record in records if record.kind == "tool_operation")
        reference = native_tool_result_reference(decode_envelope(record.data).value["payload"]["fields"]["result_payload"])
        assert reference is not None
        report["original_payload_bytes"] = payload.size
        report["retained_template_bytes"] = len(reference.template.encode("utf-8"))
        report["tool_owner_json_bytes"] = len(canonical_json_bytes(record.data))
    async with Persistence.open("native-result", tenant_id="tenant", storage=config, read_only=True) as state:
        engine = state.checkpoint.tools.state_store.storage_group.context.engine.sync_engine
        event.listen(engine, "before_cursor_execute", sql)
        try:
            counts.clear()
            loaded = await state.checkpoint.tools.get_operation(operation.tool_operation_id, tenant_id="tenant")
            assert loaded.result_payload == payload
            report["cold_tool_read"] = dict(counts)
            counts.clear()
            values = await state.checkpoint.tools.get_by_call_ids("run", ("call",), tenant_id="tenant")
            assert values[0].result_payload == payload
            report["warm_tool_batch_read"] = dict(counts)
        finally:
            event.remove(engine, "before_cursor_execute", sql)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
