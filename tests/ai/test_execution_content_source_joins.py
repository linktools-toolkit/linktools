#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Projection joins preserve actual source identity, cold retries, and narrow reads."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event
from sqlalchemy.dialects import mysql, postgresql, sqlite

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence
from linktools.ai.persistence._archive import ContentArchive, archive_content_scope, archive_scope_key
from linktools.ai.persistence._content_reference_sql import content_reference_batch_size, content_reference_statement
from linktools.ai.persistence.codec import decode_envelope, encode_envelope
from linktools.ai.storage import StoredPayload

from .test_execution_result_read_io import _seed
from .test_incremental_projection_boundaries import _storage

pytestmark = pytest.mark.asyncio
_NAMESPACE = "source-join"
_PATHS = {
    "policy": ("runtime_policy",),
    "binding": ("binding",),
    "template": ("result", "fields", "content"),
    "output": ("result", "fields", "output", "fields", "content"),
}


@pytest.mark.parametrize("target", tuple(_PATHS))
@pytest.mark.parametrize("corruption", ("missing", "body", "scope"))
async def test_projection_join_rejects_missing_damaged_and_foreign_sources(
    tmp_path: Path, target: str, corruption: str,
) -> None:
    state = Persistence(_storage("sqlite", tmp_path))
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        await _seed(state)
        repository = state.execution.executions
        owner = await repository.state_store.read_record(repository.execution_record_key("execution"))
        assert owner is not None
        value = deepcopy(dict(decode_envelope(owner.data).value))
        reference = value["payload"]["fields"]
        for name in _PATHS[target]:
            reference = reference[name]
        if corruption == "scope":
            reference["content_scope"] = archive_content_scope("foreign", "tenant")
            assert await repository.state_store.mutate(lambda transaction: transaction.replace_record(
                replace(owner, data=encode_envelope(value), storage_revision=owner.storage_revision + 1),
                expected_storage_revision=owner.storage_revision,
            ))
        else:
            scope, store = await repository.state_store.storage_group.open_archive()
            archive = ContentArchive(store, scope, namespace=_NAMESPACE, tenant_id="tenant", object_store=None)
            identity = next(item for name, item in reference.items() if name.startswith("$"))
            key, = archive.content_identity_keys((identity,))
            body = await store.read_record(key)
            assert body is not None
            if corruption == "missing":
                assert await store.mutate(lambda transaction: transaction.delete_record(
                    key, expected_storage_revision=body.storage_revision,
                ))
            else:
                assert await store.mutate(lambda transaction: transaction.replace_record(
                    replace(body, payload_bytes=b"damaged", storage_revision=body.storage_revision + 1),
                    expected_storage_revision=body.storage_revision,
                ))
        read = repository.get_view_fresh if target in {"binding", "template"} else repository.get_result_record
        with pytest.raises(AIError) as caught:
            await read("execution", tenant_id="tenant")
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()


async def test_cold_projection_join_restarts_with_the_current_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with Persistence.open(_NAMESPACE, storage=_storage("sqlite", tmp_path), tenant_id="tenant") as source:
        await _seed(source)
    async with Persistence.open(_NAMESPACE, storage=_storage("sqlite", tmp_path), tenant_id="tenant") as writer:
        async with Persistence.open(_NAMESPACE, storage=_storage("sqlite", tmp_path), tenant_id="tenant", read_only=True) as reader:
            repository = reader.execution.executions
            assert repository.state_store.storage_group.opened_archive is None
            initialize = repository._execution_content.initialize
            removed = False

            async def remove_owner_after_open(*, create: bool, share_content: bool = True) -> None:
                nonlocal removed
                await initialize(create=create, share_content=share_content)
                if not removed:
                    removed = True
                    await writer.execution.executions.state_store.mutate(lambda transaction: transaction.delete_record(
                        writer.execution.executions.execution_record_key("execution"),
                    ))

            monkeypatch.setattr(repository._execution_content, "initialize", remove_owner_after_open)
            assert await repository.get_result_record("execution", tenant_id="tenant", fresh=True) is None
            assert removed


async def test_projection_join_uses_indexed_owners_and_counts_ctes_as_reads(tmp_path: Path) -> None:
    async with Persistence.open(_NAMESPACE, storage=_storage("sqlite", tmp_path), tenant_id="tenant") as state:
        expected = await _seed(state, output=StoredPayload.inline_json("large output " * 20000))
        repository = state.execution.executions
        group = repository.state_store.storage_group
        engine = group.context.engine
        statements = []

        def capture(_connection: object, _cursor: object, statement: str, parameters: object,
                    _context: object, _many: bool) -> None:
            if statement.lstrip().upper().startswith(("SELECT", "WITH")):
                statements.append((statement, parameters))

        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        try:
            result = await repository.get_result_record("execution", tenant_id="tenant", fresh=True)
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", capture)
        assert result is not None and result.result == expected.result
        assert len(statements) == 1
        async with engine.connect() as connection:
            plan = await connection.exec_driver_sql("EXPLAIN QUERY PLAN " + statements[0][0], statements[0][1])
            details = tuple(row[-1] for row in plan)
        assert any("SEARCH ai_state_records" in row and "key_locator" in row for row in details)
        assert any("SEARCH content_source" in row and "sort_prefix" in row for row in details)
        assert any("SEARCH ai_step_messages" in row and "sha256" in row for row in details)
        _, content = await group.open_archive()
        paths = tuple(("value", "payload", "fields", *path, marker) for path, marker in (
            (("runtime_policy",), "$runtime_policy"),
            (("result", "fields", "content"), "$execution_payload"),
        ))
        for dialect in (sqlite.dialect(), postgresql.dialect(), mysql.dialect(server_version_info=(5, 7, 44))):
            query = content_reference_statement(group.metadata, dialect=dialect.name,
                owner_store_key=repository.state_store.store_key,
                owner_keys=(repository.execution_record_key("execution"),), content_store_key=content.store_key,
                content_scope_key=archive_scope_key(_NAMESPACE, "tenant"), content_paths=paths, typed_payloads=True)
            compiled = query.compile(dialect=dialect, compile_kwargs={"render_postcompile": True})
            assert (len(compiled.params) if compiled.positiontup is None else len(compiled.positiontup)) <= 999
            if dialect.name == "mysql":
                assert not str(compiled).lstrip().startswith("WITH")


async def test_content_join_batch_bound_covers_all_dialect_path_expansions(tmp_path: Path) -> None:
    async with Persistence.open(_NAMESPACE, storage=_storage("sqlite", tmp_path), tenant_id="tenant") as state:
        await _seed(state)
        repository = state.execution.executions
        group = repository.state_store.storage_group
        _, content = await group.open_archive()
        for path_count in (1, 4, 8, 16):
            keys = tuple(repository.execution_record_key(str(index))
                         for index in range(content_reference_batch_size(path_count)))
            paths = tuple(("value", "payload", "fields", "declared", str(index), "$execution_payload")
                          for index in range(path_count))
            for dialect in (sqlite.dialect(), postgresql.dialect(), mysql.dialect(server_version_info=(5, 7, 44))):
                query = content_reference_statement(group.metadata, dialect=dialect.name,
                    owner_store_key=repository.state_store.store_key, owner_keys=keys,
                    content_store_key=content.store_key, content_scope_key=archive_scope_key(_NAMESPACE, "tenant"),
                    content_paths=paths, typed_payloads=True)
                compiled = query.compile(dialect=dialect, compile_kwargs={"render_postcompile": True})
                count = len(compiled.params) if compiled.positiontup is None else len(compiled.positiontup)
                assert count <= 999


async def test_transaction_execution_read_batches_fixed_content(tmp_path: Path) -> None:
    """Terminal callers need one coherent owner/content read, not slot-by-slot loads."""
    async with Persistence.open(_NAMESPACE, storage=_storage("sqlite", tmp_path), tenant_id="tenant") as state:
        await _seed(state)
        repository = state.execution.executions
        expected = await repository.get("execution", tenant_id="tenant")
        statements: list[str] = []

        def observe(connection, cursor, statement, parameters, context, executemany):
            if statement.lstrip().upper().startswith(("SELECT", "WITH")):
                statements.append(statement)

        async def read(transaction):
            actual = await repository.get_in_transaction(transaction, "execution", tenant_id="tenant")
            assert actual == expected
            first_reads = len(statements)
            assert await repository.get_in_transaction(transaction, "execution", tenant_id="tenant") == actual
            assert len(statements) == first_reads

        engine = repository.state_store.storage_group.context.engine.sync_engine
        event.listen(engine, "before_cursor_execute", observe)
        try:
            await repository.state_store.mutate(read)
        finally:
            event.remove(engine, "before_cursor_execute", observe)
        assert len(statements) == 1
        assert "ai_state_records" in statements[0]
        assert "ai_step_messages" in statements[0]
