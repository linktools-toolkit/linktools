#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed Native captures preserve source validation and transaction boundaries."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from sqlalchemy import event, text
from sqlalchemy.dialects import mysql, postgresql, sqlite

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.persistence._archive import archive_scope_key
from linktools.ai.persistence._content_reference_sql import content_reference_batch_size, content_reference_statement
from linktools.ai.persistence._step_schema import build_step_sql_metadata
from linktools.ai.persistence.codec import decode_envelope, encode_envelope, native_text_reference
from linktools.ai.persistence.schema import build_runtime_sql_metadata
from linktools.ai.persistence.state import record_key
from linktools.ai.steps import native_store_key
from .test_native_text_result_runtime import _prepare, _publish


@pytest.mark.parametrize("backend", ("mysql", "postgresql", "sqlite"))
def test_native_capture_bind_budget_and_types(backend: str) -> None:
    metadata = build_runtime_sql_metadata(frozenset(PersistenceDomain))
    build_step_sql_metadata(metadata=metadata)
    paths = tuple(("value", "payload", "fields", name, "fields", "content", "$execution_payload")
                  for name in ("arguments_payload", "result_payload", "error_payload"))
    native_paths = tuple(("value", "payload", "fields", name, "fields", "content", marker)
        for name, marker in (("arguments_payload", "$native_tool_arguments"), ("result_payload", "$native_tool_result")))
    size = content_reference_batch_size(len(paths), native_path_count=len(native_paths))
    statement = content_reference_statement(metadata, dialect=backend, owner_store_key=b"owner",
        owner_keys=tuple(record_key("namespace", "tenant", "checkpoint", "tool_operation", (str(i),)) for i in range(size)),
        content_store_key=b"archive", content_scope_key=archive_scope_key("namespace", "tenant"),
        content_paths=paths, typed_payloads=True, native_root_paths=native_paths)
    dialect = {"mysql": mysql, "postgresql": postgresql, "sqlite": sqlite}[backend].dialect()
    compiled = statement.compile(dialect=dialect, compile_kwargs={"render_postcompile": True})
    assert len(compiled.positiontup or compiled.params) <= 999
    assert "COLLATE utf8mb4_bin)" not in str(compiled)
    assert "fixed_native_nodes" in str(compiled)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ("singleton", "head", "nonhead", "scope", "missing", "bytes"))
async def test_fixed_native_capture_remains_exact_and_transaction_scoped(tmp_path: Path, case: str) -> None:
    config = PersistenceStorage.sqlite(tmp_path / "native.db")
    async with Persistence.open("native-result", tenant_id="tenant", storage=config) as state:
        commands, commit, current, checkpoint = await _prepare(state, "exact output", "exact output")
        if case != "singleton":
            checkpoint = replace(checkpoint, messages=[ModelResponse(parts=[TextPart("exact output")]) for _ in range(4)])
        await _publish(commands, commit, current, checkpoint)
        repository = state.execution.executions
        stored = await repository.state_store.read_record(repository.execution_record_key("execution"))
        value = deepcopy(decode_envelope(stored.data).value)
        content = value["payload"]["fields"]["result"]["fields"]["output"]["fields"]["content"]
        source = native_text_reference(value["payload"]["fields"]["result"]["fields"]["output"])
        assert source is not None
        assert source.root.slices[0].count == (1 if case == "singleton" else 4)
        if case == "nonhead":
            content["$native_text"]["slices"][0].update(start=0, end=1)
            await repository.state_store.mutate(lambda tx: tx.replace_record(
                replace(stored, data=encode_envelope(value), storage_revision=stored.storage_revision + 1),
                expected_storage_revision=stored.storage_revision))
    async with Persistence.open("native-result", tenant_id="tenant", storage=config) as state:
        repository = state.execution.executions
        group = repository.state_store.storage_group
        # Real first cold read discovers archive and validates the fixed source.
        loaded = await repository.get_fresh("execution", tenant_id="tenant")
        assert loaded.result == commit.result
        statements: list[str] = []
        def observe(connection: object, cursor: object, statement: str, parameters: object,
                    context: object, executemany: bool) -> None:
            if statement.lstrip().upper().startswith(("SELECT", "WITH")):
                statements.append(statement)
        event.listen(group.context.engine.sync_engine, "before_cursor_execute", observe)
        try:
            assert await repository.get_fresh("execution", tenant_id="tenant") == loaded
        finally:
            event.remove(group.context.engine.sync_engine, "before_cursor_execute", observe)
        assert len(statements) == (2 if case == "nonhead" else 1)
        if case in {"singleton", "head", "nonhead"}:
            return
        if case == "scope":
            from sqlalchemy import update
            from linktools.ai.core import canonical_json_bytes
            stored = await repository.state_store.read_record(repository.execution_record_key("execution"))
            value = deepcopy(decode_envelope(stored.data).value)
            value["payload"]["fields"]["result"]["fields"]["output"]["fields"]["content"]["$native_text"]["group_id"] = "other"
            table = group.metadata.tables["ai_state_records"]
            async with group.context.engine.begin() as connection:
                await connection.execute(update(table).where(table.c.key == stored.key).values(
                    payload_json=canonical_json_bytes(encode_envelope(value)).decode("utf-8")))
        else:
            async with group.context.engine.begin() as connection:
                if case == "missing":
                    await connection.execute(text("DELETE FROM ai_step_sequences WHERE store_key = :store"),
                        {"store": native_store_key("native-result", "tenant")})
                else:
                    await connection.execute(text("UPDATE ai_step_messages SET message = :bad WHERE store_key = :store"),
                        {"bad": b"broken source", "store": native_store_key("native-result", "tenant")})
        # A new transaction cannot reuse the successful earlier source capture.
        with pytest.raises(AIError) as damaged:
            await repository.get_fresh("execution", tenant_id="tenant")
        assert damaged.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
