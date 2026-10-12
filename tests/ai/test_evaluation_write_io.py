#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batched publication preserves authorization, retention, and transaction truth."""

import asyncio
import getpass
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import event
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from linktools.ai.core import service_principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import CaseRef, CaseSpec, DatasetRef, DatasetSpec
from linktools.ai.migrate import provision_database
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import Runtime, RuntimeContext, RuntimeStorage


_PRINCIPAL = service_principal("evaluation-io", "owner")


def _dataset(name: str, count: int, *, revision: int = 1) -> DatasetSpec:
    return DatasetSpec(DatasetRef(name, revision), tuple(
        CaseSpec.task(CaseRef(name, str(index), 1), input={"answer": index})
        for index in range(count)
    ))


async def _assert_publication(runtime: Runtime, peer: Runtime, state: RuntimeStorage) -> None:
    original = _dataset("atomic", 1)
    await runtime.evaluations.publish_dataset(original, principal=_PRINCIPAL, idempotency_key="first")
    extended = DatasetSpec(DatasetRef("atomic", 2), (
        *(_dataset("atomic", 71).cases[1:]),
        CaseSpec.task(original.cases[0].ref, input={"answer": "changed"}),
    ))
    with pytest.raises(AIError) as conflict:
        await runtime.evaluations.publish_dataset(extended, principal=_PRINCIPAL, idempotency_key="atomic")
    assert conflict.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
    assert await state.evaluation.records.get_case(extended.cases[0].ref) is None
    assert await state.evaluation.records.get_dataset(extended.ref) is None
    corrected = DatasetSpec(extended.ref, (*extended.cases[:-1], original.cases[0]))
    assert await runtime.evaluations.publish_dataset(corrected, principal=_PRINCIPAL, idempotency_key="atomic") == corrected.ref

    concurrent = _dataset("concurrent", 8)
    results = await asyncio.gather(*(
        owner.evaluations.publish_dataset(concurrent, principal=_PRINCIPAL, idempotency_key="concurrent")
        for owner in (runtime, peer)
    ), return_exceptions=True)
    assert any(result == concurrent.ref for result in results)
    assert all(result == concurrent.ref or (
        isinstance(result, AIError) and result.code is ErrorCode.STORAGE_CONFLICT
    ) for result in results)
    for owner in (runtime, peer):
        assert await owner.evaluations.publish_dataset(concurrent, principal=_PRINCIPAL,
                                                       idempotency_key="concurrent") == concurrent.ref
    page = await peer.evaluations.list_cases(corrected.ref, principal=_PRINCIPAL, limit=100)
    assert tuple(value.ref for value in page.items) == tuple(value.ref for value in corrected.cases)

    retained = _dataset("retained", 1)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
    await runtime.evaluations.publish_dataset(retained, principal=_PRINCIPAL,
                                             idempotency_key="retained", content_expires_at=expires_at)
    assert await state.evaluation.records.purge_expired_datasets(
        now=expires_at + timedelta(seconds=1), owner_principal_id=_PRINCIPAL.principal_id, limit=10,
    ) == (retained.ref,)
    assert await state.evaluation.records.dataset_owner(retained.ref) == _PRINCIPAL.principal_id
    with pytest.raises(AIError) as unavailable:
        await peer.evaluations.publish_dataset(retained, principal=_PRINCIPAL, idempotency_key="resurrect")
    assert unavailable.value.code is ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE
    other = service_principal(_PRINCIPAL.tenant_id, "other")
    with pytest.raises(AIError) as denied:
        await peer.evaluations.publish_dataset(retained, principal=other, idempotency_key="steal")
    assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED


async def _assert_sql_publication(engine: AsyncEngine) -> None:
    await provision_database(engine)
    state, peer_state = RuntimeStorage.sql(engine), RuntimeStorage.sql(engine)
    context = RuntimeContext(None, tenant_id=_PRINCIPAL.tenant_id)
    async with Runtime.open("evaluation-io", models=ModelRegistry(), storage=state, context=context) as runtime:
        async with Runtime.open("evaluation-io", models=ModelRegistry(), storage=peer_state, context=context) as peer:
            statements: list[tuple[str, bool]] = []
            large_parameters: list[int] = []
            def capture(conn, cursor, sql, params, context, many):
                if sql.split()[0] in {"SELECT", "INSERT", "UPDATE", "DELETE"}:
                    statements.append((sql.split()[0], many))
                if sql.startswith("INSERT"):
                    values = params.values() if isinstance(params, Mapping) else params
                    large_parameters.append(sum(
                        isinstance(value, (str, bytes)) and len(value) > 64 * 1024
                        for value in values
                    ))
            event.listen(engine.sync_engine, "before_cursor_execute", capture)
            try:
                spec = _dataset("counted", 32)
                await runtime.evaluations.publish_dataset(spec, principal=_PRINCIPAL, idempotency_key="counted")
                assert not any(many for _, many in statements)
                counts = Counter(command for command, _ in statements)
                assert counts["SELECT"] <= 3 * len(spec.cases)
                assert 0 < counts["INSERT"] <= 4, "publication must not insert once per case"
                assert counts["UPDATE"] == counts["DELETE"] == 0
                statements.clear()
                await runtime.evaluations.publish_dataset(spec, principal=_PRINCIPAL, idempotency_key="counted")
                counts = Counter(command for command, _ in statements)
                assert counts["SELECT"] <= 3 * len(spec.cases)
                assert counts["INSERT"] == counts["UPDATE"] == counts["DELETE"] == 0
                large = DatasetSpec(DatasetRef("large", 1), tuple(
                    CaseSpec.task(CaseRef("large", str(index), 1), input={"text": "x" * 70000})
                    for index in range(2)
                ))
                statements.clear()
                large_parameters.clear()
                await runtime.evaluations.publish_dataset(large, principal=_PRINCIPAL, idempotency_key="large")
                assert large_parameters and max(large_parameters) <= 1
                for case in large.cases:
                    restored = await state.evaluation.records.get_case(case.ref)
                    assert restored.input.input["text"] == "x" * 70000
            finally:
                event.remove(engine.sync_engine, "before_cursor_execute", capture)
            await _assert_publication(runtime, peer, state)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_publication_batches_preserve_backend_atomicity(tmp_path: Path, backend: str) -> None:
    if backend == "sqlite":
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'state.db'}")
        try:
            await _assert_sql_publication(engine)
        finally:
            await engine.dispose()
        return
    state = (RuntimeStorage.in_memory() if backend == "memory"
             else RuntimeStorage.filesystem(tmp_path))
    context = RuntimeContext(None, tenant_id=_PRINCIPAL.tenant_id)
    async with Runtime.open("evaluation-io", models=ModelRegistry(), storage=state, context=context) as runtime:
        await _assert_publication(runtime, runtime, state)
    if backend == "filesystem":
        async with Runtime.open("evaluation-io", models=ModelRegistry(),
                                storage=RuntimeStorage.filesystem(tmp_path), context=context) as reopened:
            cases = await reopened.evaluations.list_cases(DatasetRef("atomic", 2), principal=_PRINCIPAL)
            assert len(cases.items) == 71


@pytest.mark.asyncio
async def test_server_publication_batches_preserve_sql_and_replay(_server: tuple[str, list[str]]) -> None:
    name, command = _server
    if name == "postgresql":
        url = URL.create("postgresql+asyncpg", username=getpass.getuser(), database="postgres",
                         query={"host": command[command.index("-h") + 1]})
    else:
        socket = next(value.split("=", 1)[1] for value in command if value.startswith("--socket="))
        url = URL.create("mysql+asyncmy", username="root", database="metrics_test",
                         query={"unix_socket": socket})
    engine = create_async_engine(url, isolation_level="READ COMMITTED")
    try:
        await _assert_sql_publication(engine)
    finally:
        await engine.dispose()
