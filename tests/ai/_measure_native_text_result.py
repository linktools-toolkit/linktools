#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Manually measure native terminal text ownership; pass a checkout root."""

import asyncio
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile


async def main(root: Path) -> None:
    sys.path[:0] = [str(root), str(root / "linktools-ai/src"), str(root / "linktools/src")]
    from sqlalchemy import event
    from linktools.ai.core import ExecutionStatus
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.persistence import Persistence, PersistenceStorage
    from linktools.ai.persistence.codec import decode_envelope, execution_payload_slots, native_text_reference
    from linktools.ai.storage import StoredPayload
    from tests.ai.test_native_text_result_runtime import _prepare, _publish

    measurements = {}
    counts: Counter[str] = Counter()

    def capture(_connection: object, _cursor: object, statement: str,
                _parameters: object, _context: object, _many: bool) -> None:
        operation = statement.split()[0]
        counts["SELECT" if operation == "WITH" else operation] += 1

    with tempfile.TemporaryDirectory(prefix="native-result-") as temporary:
        config = PersistenceStorage.sqlite(Path(temporary) / "results.db")
        async with Persistence.open("native-result", tenant_id="tenant", storage=config) as state:
            commands, commit, current, checkpoint = await _prepare(state, "source " * 12000, "source " * 12000)
            repository = state.execution.executions
            engine = repository.state_store.storage_group.context.engine.sync_engine
            event.listen(engine, "before_cursor_execute", capture)
            try:
                await _publish(commands, commit, current, checkpoint)
            finally:
                event.remove(engine, "before_cursor_execute", capture)
            measurements["terminal_publication"] = dict(counts)
            raw = await repository.state_store.read_record(repository.execution_record_key("execution"))
            reference = next(reference for _, payload in execution_payload_slots(decode_envelope(raw.data).value["payload"])
                             if (reference := native_text_reference(payload)) is not None)
            measurements["payload_json_bytes"] = commit.result.output.size
            other = replace(current, execution_id="other", root_execution_id="other")
            await repository.create_execution(other)
            output = StoredPayload.inline_json("different " * 12000)
            forged = replace(reference, digest=output.digest, size=output.size)
            candidate = replace(commit, execution=replace(other, status=ExecutionStatus.SUCCEEDED),
                                result=replace(commit.result, output=output))
            try:
                await repository.state_store.mutate(lambda transaction: repository.commit_terminal_in_transaction(
                    transaction, candidate, result_source=forged))
            except AIError as error:
                assert error.code is ErrorCode.STORAGE_INTEGRITY_ERROR
                measurements["wrong_existing_source"] = error.code.value
            else:
                raise AssertionError("writer accepted different stored source text")
            assert (await repository.get("other", tenant_id="tenant")).status is ExecutionStatus.STARTED
        async with Persistence.open("native-result", tenant_id="tenant", storage=config, read_only=True) as cold:
            repository = cold.execution.executions
            engine = repository.state_store.storage_group.context.engine.sync_engine
            event.listen(engine, "before_cursor_execute", capture)
            try:
                for phase in ("cold_result", "warm_result"):
                    counts.clear()
                    result = await repository.get_result_record("execution", tenant_id="tenant", fresh=True)
                    assert result.result == commit.result
                    measurements[phase] = dict(counts)
            finally:
                event.remove(engine, "before_cursor_execute", capture)
            assert (await repository.get("execution", tenant_id="tenant")).result == commit.result
            assert (await repository.get_timeline_many(("execution",), tenant_id="tenant"))["execution"].result == commit.result
        print(json.dumps(measurements, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]).resolve()))
