#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit model identities share reads without enumerating other executions."""

from dataclasses import replace

import pytest

from linktools.ai.core import Principal, ResourceKind, ResourceRef
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.history._service import DefaultExecutionHistoryService

from .test_model_interaction_lifecycle_paging import _reader, _interaction

pytestmark = pytest.mark.asyncio


async def test_exact_batch_order_deduplication_and_no_descendant_discovery() -> None:
    reader, executions, store = _reader()
    async def no_children(*args, **kwargs):
        raise AssertionError("exact reads must not enumerate descendants")
    executions.list_children = no_children
    values = await reader.read_model_interactions(
        "root", tenant_id="tenant", identities=((1, 2), (1, 1), (1, 2)),
    )
    assert [value.model_request_seq for value in values] == [2, 1]
    assert all(not value.content_included for value in values)
    assert await reader.read_model_interactions("root", tenant_id="tenant", identities=()) == ()


@pytest.mark.parametrize("identities", (((0, 1),), ((1, True),), ((1, 0),), ((1,),), ((1, 1),) * 201))
async def test_exact_batch_rejects_invalid_keys(identities) -> None:
    reader, _, _ = _reader()
    with pytest.raises(AIError) as error:
        await reader.read_model_interactions("root", tenant_id="tenant", identities=identities)
    assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID


@pytest.mark.parametrize("fault", ("missing", "foreign", "duplicate"))
async def test_exact_batch_distinguishes_missing_and_corrupt_records(fault: str) -> None:
    reader, _, store = _reader()
    run_id = next(iter(store.runs))
    if fault == "missing":
        store.interactions[run_id] = [_interaction(run_id, 5)]
    elif fault == "foreign":
        store.interactions[run_id][0] = replace(store.interactions[run_id][0], agent_run_id="foreign")
    else:
        store.interactions[run_id][1] = store.interactions[run_id][0]
    with pytest.raises(AIError) as error:
        await reader.read_model_interactions("root", tenant_id="tenant", identities=((1, 1), (1, 2)))
    assert error.value.code is (ErrorCode.STORAGE_NOT_FOUND if fault == "missing" else ErrorCode.STORAGE_INTEGRITY_ERROR)


async def test_exact_batch_resolves_content_once_per_run_outside_metadata_scope() -> None:
    reader, _, store = _reader()
    run_id = next(iter(store.runs))
    store.interactions[run_id] = [_interaction(run_id, sequence) for sequence in (1, 2)]
    in_scope = False
    resolved = []
    async def scoped(executions, callback, *, include_content=False):
        assert include_content
        nonlocal in_scope
        in_scope = True
        try:
            return await callback()
        finally:
            in_scope = False
    async def pending(*args, **kwargs):
        assert not in_scope
        return None
    async def resolve(values, **kwargs):
        assert not in_scope
        resolved.append(len(values))
        return [((), None, b"{}", None, None) for value in values]
    reader._read_model_metadata = scoped
    reader.bind_pending_execution_reader(pending)
    store.resolve_model_interactions = resolve
    values = await reader.read_model_interactions(
        "root", tenant_id="tenant", identities=((1, 2), (1, 1)), include_content=True,
    )
    assert len(values) == 2 and all(value.content_included for value in values)
    assert resolved == [2]


async def test_exact_batch_keeps_service_authorization_before_reader_access() -> None:
    reader, executions, _ = _reader()
    class Deny:
        async def authorize(self, *args):
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
    async def forbidden(*args, **kwargs):
        raise AssertionError("denied query reached reader")
    async def header(execution_id, **kwargs):
        return ResourceRef(ResourceKind.EXECUTION, execution_id, "tenant", "owner")
    executions.get_header = header
    reader.read_model_interactions = forbidden
    service = DefaultExecutionHistoryService(executions, Deny(), reader)
    with pytest.raises(AIError) as error:
        await service.read_model_interactions("root", principal=Principal("owner", "tenant"), identities=((1, 1),))
    assert error.value.code is ErrorCode.AUTHORIZATION_DENIED


@pytest.mark.parametrize("backend", ("memory", "local_sqlite", "sqlite"))
async def test_exact_batch_real_archive_and_sql_read_reduction(tmp_path, backend) -> None:
    from datetime import datetime, timezone
    from sqlalchemy import event
    from sqlalchemy.ext.asyncio import create_async_engine
    from linktools.ai.core import ExecutionStatus, HmacCursorSigner, agent_conversation_id, agent_run_id
    from linktools.ai.history import StepExecutionHistoryReader
    from linktools.ai.migrate import provision_database
    from linktools.ai.persistence import Persistence, PersistenceStorage, PersistenceDomain
    from linktools.ai.persistence.runs import AgentRunRecord, StagedModelInteraction
    from .test_history_projection_conformance import _record
    from .test_incremental_projection_boundaries import _storage
    engine = None
    if backend == "sqlite":
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'batch.db'}")
        await provision_database(engine)
        storage = PersistenceStorage.sql(engine)
    else:
        storage = _storage(backend, tmp_path)
    state = Persistence(storage)
    statements = []
    def capture(*args):
        statements.append(args[2])
    await state.initialize(namespace="history", tenant_id="tenant")
    try:
        await state.execution.executions.create_execution(_record(ExecutionStatus.STARTED, 1))
        run_id = agent_run_id(namespace="history", tenant_id="tenant", execution_id="execution", agent_run_seq=1)
        await state.run_store.register_agent_run(AgentRunRecord(
            run_id, agent_conversation_id=agent_conversation_id(namespace="history", tenant_id="tenant", execution_id="execution"),
            agent_id="default", metadata={"agent_run_seq": "1"},
        ), execution_id="execution")
        for sequence in range(1, 5):
            state.run_store.stage_model_interaction(StagedModelInteraction(
                run_id, sequence, sequence, "agent", None, {}, None, None, None,
                "RUNNING", None, None, None, datetime.now(timezone.utc), None,
            ))
        await state.run_store.flush_execution_projection(run_id, execution_id="execution")
        reader = StepExecutionHistoryReader(
            namespace="history", executions=state.execution.executions,
            store=state.run_store.read_store(PersistenceDomain.TRACE),
            cursor_signer=HmacCursorSigner("test", b"test"),
        )
        if engine is not None:
            event.listen(engine.sync_engine, "before_cursor_execute", capture)
        singles = []
        for sequence in range(1, 5):
            singles.extend(await reader.read_model_interaction_metadata(
                "execution", tenant_id="tenant", agent_run_seq=1,
                after_model_request_seq=sequence-1, through_model_request_seq=sequence,
            ))
        before = len(statements)
        statements.clear()
        batch = await reader.read_model_interactions(
            "execution", tenant_id="tenant", identities=tuple((1, sequence) for sequence in range(1, 5)),
        )
        after = len(statements)
        assert batch == tuple(singles)
        if engine is not None:
            assert after < before
            print(f"SQL exact metadata: four singleton reads={before}, batch={after}")
        bodies = await reader.read_model_interactions(
            "execution", tenant_id="tenant", identities=((1, 3), (1, 1)), include_content=True,
        )
        assert [value.model_request_seq for value in bodies] == [3, 1]
        assert all(value.content_included for value in bodies)
    finally:
        if engine is not None:
            event.remove(engine.sync_engine, "before_cursor_execute", capture)
        await state.close()
        if engine is not None:
            await engine.dispose()


async def test_exact_batch_http_content_and_permission_parity() -> None:
    from linktools.ai.capability import ExecutionCapability, TraceCapability
    from linktools.ai.runtime import Runtime, RuntimeFeatures
    from linktools.ai.persistence import Persistence, PersistenceStorage
    from linktools.ai.web import create_app
    from ._runtime_test_helpers import RuntimeUsageModels, model_features
    from .test_web_console import client
    features = model_features(RuntimeUsageModels(), RuntimeFeatures("batch-http"))
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.capability(TraceCapability(), scope="runtime")
    async with Persistence.open("batch-http", storage=PersistenceStorage.in_memory()) as state:
        async with Runtime.attach(persistence=state, features=features) as runtime:
            execution = await runtime.agents.get().start("hello")
            await execution.wait()
            async with client(create_app(runtime=runtime)) as http:
                path = f"/api/executions/{execution.execution_id}/models-exact"
                response = await http.get(path + "?identity=1:1&identity=1:1&include_content=true")
                assert response.status_code == 200, response.text
                items = response.json()["items"]
                assert len(items) == 1 and items[0]["content_included"]
                assert items[0]["response"]["parts"][0]["content"] == "done"
                assert (await http.get(path + "?identity=invalid")).status_code == 400
                denied = await http.get("/api/executions/missing/models-exact?identity=1:1")
                assert denied.status_code == 404 and denied.json()["code"] == "AUTHORIZATION_DENIED"


async def test_exact_batch_web_grouping_failure_isolation_and_stale_work() -> None:
    import shutil
    import subprocess
    from pathlib import Path
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed for Web contract tests")
    root = Path(__file__).parents[2]
    result = subprocess.run([
        node, str(Path(__file__).with_name("web_history_exact_batch.mjs")),
        str(root / "linktools-ai/src/linktools/ai/web/assets/app.js"),
    ], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
