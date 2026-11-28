#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real Runtime producers keep immutable execution bodies out of control rows."""

import asyncio
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo
from pydantic_ai_harness.step_persistence import SqliteStepStore
from sqlalchemy import event

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import CommitPolicy, ExecutionStatus, TenantAuthorizationPolicy, canonical_json_bytes
from linktools.ai.persistence import NativeCheckpoint, Persistence, PersistenceStorage
from linktools.ai.persistence._archive import ContentArchive
from linktools.ai.model import StoredUserInput
from linktools.ai.persistence.codec import decode_domain, decode_envelope, execution_payload_slots, payload_reference
from linktools.ai.persistence.contracts import ExecutionRecord, RecoveryHandoffPhase
from linktools.ai.persistence.state import StoredRecord
from linktools.ai.runtime import Runtime, RuntimeFeatures

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features, native_checkpoint
from ._session_tool_test_helpers import _split_sqlite_storage


_INPUT = "large immutable request body " * 4000
_OUTPUT = "large immutable answer body " * 4000
_ROW_LIMIT = 16 * 1024
_INPUT_SLOT = ("stored_user_input",)
_OUTPUT_SLOT = ("result", "fields", "output")
_RESULT_SLOT = ("result",)
_HANDOFF_TEMPLATE_SLOT = ("recovery", "fields", "terminal_handoff")
_HANDOFF_SLOT = ("recovery", "fields", "terminal_handoff", "fields", "output")


class _HostAuthorization(TenantAuthorizationPolicy):
    """Explicit host authorization for the canonical SQL session tests."""


def _payload_refs(record: StoredRecord) -> dict[tuple[str, ...], tuple[str, str, str]]:
    payload = decode_envelope(record.data).value["payload"]
    return {path: reference for path, value in execution_payload_slots(payload)
            if (reference := payload_reference(value)) is not None}


@contextmanager
def _execution_sql_bindings(persistence: Persistence) -> Iterator[list[dict]]:
    """Observe bytes actually bound to SQL, including intermediate control CAS."""
    engine = persistence.execution.executions.state_store.context.engine.sync_engine
    rows: list[dict] = []

    def capture(connection, cursor, statement, parameters, execution_context, many) -> None:
        if "ai_execution_records" not in statement or not statement.lstrip().upper().startswith(("INSERT", "UPDATE")):
            return
        batches = parameters if many else (parameters,)
        for batch in batches:
            values = batch.values() if isinstance(batch, Mapping) else batch
            for value in values:
                if not isinstance(value, str) or "execution_record" not in value:
                    continue
                envelope = json.loads(value)
                if isinstance(envelope, str):
                    envelope = json.loads(envelope)
                payload = decode_envelope(envelope).value["payload"]
                if payload.get("$dataclass") == "execution_record":
                    # Bodies must never be rebound by a metadata-only UPDATE.
                    assert len(value.encode("utf-8")) < _ROW_LIMIT
                    assert _INPUT not in value and _OUTPUT not in value
                    rows.append(payload)

    event.listen(engine, "before_cursor_execute", capture)
    try:
        yield rows
    finally:
        event.remove(engine, "before_cursor_execute", capture)


async def _compact_row(persistence: Persistence, execution_id: str) -> StoredRecord:
    repository = persistence.execution.executions
    row = await repository.state_store.read_record(repository.execution_record_key(execution_id))
    assert row is not None
    encoded = canonical_json_bytes(row.data)
    assert len(encoded) < _ROW_LIMIT
    assert _INPUT.encode() not in encoded and _OUTPUT.encode() not in encoded
    return row


async def _inspect_without_payload_reads(
    runtime: Runtime, execution_id: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_contents = ContentArchive.read_contents

    async def metadata_only(self: ContentArchive, identities: Sequence[str], *, codec: str,
                            reserve: Callable[[str, int], None] | None = None) -> tuple[bytes, ...]:
        contents = await read_contents(self, identities, codec=codec, reserve=reserve)
        if codec == "execution-payload":
            for raw in contents:
                template = json.loads(raw)
                assert template.get("$dataclass") == "result_record", "inspection hydrated an input or output body"
                assert template["fields"]["output"] is None, "result metadata duplicated its output body"
        return contents

    with monkeypatch.context() as guarded:
        guarded.setattr(ContentArchive, "read_contents", metadata_only)
        repository = runtime._state.execution.executions
        header = await repository.get_header_fresh(execution_id, tenant_id=runtime.tenant_id)
        assert header is not None
        view = await runtime.executions.inspect(execution_id, principal=runtime.default_principal)
        assert view.status is ExecutionStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_independent_execution_keeps_control_writes_compact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts = ["first small request", f"second request: {_INPUT}", f"third request: {_INPUT}"]
    calls = 0

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        visible = [part.content for message in messages for part in message.parts
                   if isinstance(part, UserPromptPart)]
        assert visible == [prompts[calls]]
        answers = [part.content for message in messages for part in message.parts if isinstance(part, TextPart)]
        assert answers == []
        calls += 1
        return ModelResponse(parts=[TextPart(_OUTPUT)])

    features = RuntimeFeatures("execution-payload-runtime")
    features.capability(ExecutionCapability(retain_history=False, commit=CommitPolicy.sync()), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    models = _Models(_UsageFunctionModel(model))
    database = tmp_path / "runtime.db"
    ids = []
    async with Persistence.open("execution-payload-runtime", storage=PersistenceStorage.sqlite(database)) as state:
        async with Runtime.open(persistence=state, features=model_features(models, features), metrics=None) as runtime:
            target = runtime.agents.get()
            with _execution_sql_bindings(state) as bindings:
                for index, prompt in enumerate(prompts):
                    execution = await target.start(prompt, idempotency_key=f"request-{index}")
                    result = (await execution.wait(timeout_seconds=10)).result
                    assert result.status is ExecutionStatus.SUCCEEDED
                    assert result.output == {"text": _OUTPUT}
                    ids.append(execution.execution_id)
                    row = await _compact_row(state, execution.execution_id)
                    refs = _payload_refs(row)
                    assert set(refs) == {_INPUT_SLOT, _RESULT_SLOT, _OUTPUT_SLOT}
                    record = await state.execution.executions.get_fresh(execution.execution_id, tenant_id=runtime.tenant_id)
                    assert record is not None
                    assert not record.conversation
                    assert not record.restart_checkpoints and not record.durable_trace
                    assert record.stored_user_input.payload.decode() == prompt
                    assert record.result.output.kind == "inline"
                    assert record.result.output.decode() == {"text": _OUTPUT}
                    await _inspect_without_payload_reads(runtime, execution.execution_id, monkeypatch)
                assert calls == len(prompts)
                for execution_id in ids:
                    writes = [row for row in bindings if row["fields"]["execution_id"] == execution_id]
                    assert len(writes) > 1, "the real producer must exercise control updates"
                    for slot in (_INPUT_SLOT, _OUTPUT_SLOT):
                        references = [value["fields"]["content"]["$execution_payload"] for write in writes
                                      for path, value in execution_payload_slots(write) if path == slot]
                        assert references and all(reference is True for reference in references)
    # A fresh Runtime cannot satisfy inspect or result reads from the producer's caches.
    async with Persistence.open("execution-payload-runtime", storage=PersistenceStorage.sqlite(database)) as state:
        async with Runtime.open(persistence=state, features=model_features(models, features), metrics=None) as runtime:
            for execution_id in (ids[0], ids[-1]):
                await _inspect_without_payload_reads(runtime, execution_id, monkeypatch)
                result = await runtime.executions.result(execution_id, principal=runtime.default_principal)
                assert result.output == {"text": _OUTPUT}
            assert calls == len(prompts)


@pytest.mark.asyncio
@pytest.mark.parametrize("large_input", (False, True), ids=("large-output", "large-input-and-output"))
async def test_native_session_retains_large_messages_without_execution_control_rows(
    tmp_path: Path, large_input: bool,
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    prompts = ["first small request", _INPUT if large_input else "second small request"]
    seen = []

    async def model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        visible = [part.content for message in messages for part in message.parts
                   if isinstance(part, UserPromptPart)]
        assert visible == prompts[:len(seen) + 1]
        seen.append(visible[-1])
        if visible[-1] == prompts[1]:
            entered.set()
            await release.wait()
        return ModelResponse(parts=[TextPart(_OUTPUT)])

    name = "native-large-turn"
    features = RuntimeFeatures(name)
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    models = _Models(_UsageFunctionModel(model))
    storage = PersistenceStorage.from_root(tmp_path / name)
    async with Persistence.open(name, storage=storage) as state:
        async with Runtime.open(
            persistence=state, features=model_features(models, features),
            authorization=_HostAuthorization(), metrics=None,
        ) as runtime:
            agent = runtime.agents.get()
            await agent.create_session("session")
            with _execution_sql_bindings(state) as bindings:
                first = await agent.start(prompts[0], session_id="session", idempotency_key="first")
                assert (await first.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
                large = await agent.start(prompts[1], session_id="session", idempotency_key="large")
                try:
                    await asyncio.wait_for(entered.wait(), 5)
                    await runtime.sessions.inspect("session", principal=runtime.default_principal)
                    active = await state.session_turns.get_turn(large.key)
                    assert active["result_json"] is None
                    if large_input:
                        request = active["request_json"]
                        assert request["prompt"] == "" and _INPUT.encode() not in canonical_json_bytes(request)
                        payload = decode_domain(request["input"], StoredUserInput)
                        assert payload.codec == "text" and payload.payload.kind == "object"
                        assert payload.payload.size == len(_INPUT.encode())
                finally:
                    release.set()
                assert (await large.wait_final(timeout_seconds=10)).view["output"] == {"text": _OUTPUT}
                assert not bindings
                for turn in (first, large):
                    assert await state.execution.executions.get(turn.key.turn_id, tenant_id=runtime.tenant_id) is None
                    row = await state.session_turns.get_turn(turn.key)
                    encoded = canonical_json_bytes(row["result_json"])
                    assert len(encoded) < _ROW_LIMIT and _OUTPUT.encode() not in encoded

    async with Runtime.open(
        name, storage=storage, features=model_features(models, features),
        authorization=_HostAuthorization(), metrics=None,
    ) as reopened:
        agent = reopened.agents.get()
        timeline = await agent.session("session").timeline()
        assert [item["user_input"]["prompt"] for item in timeline["items"]] == prompts
        assert [item["output"] for item in timeline["items"]] == [{"text": _OUTPUT}] * len(prompts)
        replay = await agent.start(prompts[1], session_id="session", idempotency_key="large")
        assert replay.key == large.key
        assert (await replay.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
        assert seen == prompts


@pytest.mark.asyncio
async def test_execution_payload_runtime_prepared_handoff_reuses_body_after_cold_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[TextPart(_OUTPUT)])

    features = RuntimeFeatures("execution-payload-handoff", defaults={"execution": {"auto_recover": True}})
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    models = _Models(_UsageFunctionModel(model))
    database = tmp_path / "handoff.db"
    storage = _split_sqlite_storage(database)
    async with Persistence.open("execution-payload-handoff", storage=storage) as state:
        async with Runtime.open(persistence=state, features=model_features(models, features), metrics=None) as owner:
            backend = owner._execution_service.runtime_backend()
            reconcile = backend._reconcile_handoff

            async def pause_handoff(checkpoint: ExecutionRecord) -> ExecutionRecord:
                assert checkpoint.recovery.handoff_phase is RecoveryHandoffPhase.PREPARED
                prepared.set()
                await release.wait()
                return await reconcile(checkpoint)

            monkeypatch.setattr(backend, "_reconcile_handoff", pause_handoff)
            with _execution_sql_bindings(state) as bindings:
                execution = await owner.agents.get().start(_INPUT, idempotency_key="handoff")
                try:
                    await asyncio.wait_for(prepared.wait(), 10)
                    row = await _compact_row(state, execution.execution_id)
                    refs = _payload_refs(row)
                    assert set(refs) == {_INPUT_SLOT, _HANDOFF_TEMPLATE_SLOT, _HANDOFF_SLOT}
                    stored = await state.execution.executions.get_fresh(execution.execution_id, tenant_id=owner.tenant_id)
                    assert stored.recovery.handoff_phase is RecoveryHandoffPhase.PREPARED
                    assert stored.recovery.terminal_handoff.outcome.output.decode() == {"text": _OUTPUT}
                    async with Persistence.open("execution-payload-handoff", storage=storage) as reader_state:
                        async with Runtime.open(persistence=reader_state,
                            features=model_features(models, features), metrics=None) as reader:
                            result = await reader.executions.result(execution.execution_id, principal=reader.default_principal)
                            assert result.status is ExecutionStatus.SUCCEEDED
                            assert result.output == {"text": _OUTPUT}
                            finished = await _compact_row(reader_state, execution.execution_id)
                            final_refs = _payload_refs(finished)
                            assert final_refs[_OUTPUT_SLOT] == refs[_HANDOFF_SLOT]
                            await _inspect_without_payload_reads(reader, execution.execution_id, monkeypatch)
                            assert calls == 1
                finally:
                    release.set()
                assert (await execution.wait(timeout_seconds=10)).result.output == {"text": _OUTPUT}
                assert bindings and calls == 1


@pytest.mark.asyncio
async def test_execution_payload_runtime_native_checkpoint_retains_body_after_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "execution-payload-checkpoint"
    features = RuntimeFeatures(name, defaults={"execution": {"auto_recover": True}})
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    step_database = tmp_path / "steps.db"
    features.capability(native_checkpoint(database=step_database), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    calls = []

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(tuple(part.content for message in messages for part in message.parts if isinstance(part, UserPromptPart)))
        return ModelResponse(parts=[TextPart(_OUTPUT)])

    storage = _split_sqlite_storage(tmp_path / "checkpoints.db")
    models = _Models(_UsageFunctionModel(model))
    async with Persistence.open(name, storage=storage) as state:
        async with Runtime.open(persistence=state, features=model_features(models, features), metrics=None) as runtime:
            with _execution_sql_bindings(state) as bindings:
                execution = await runtime.agents.get().start(_INPUT, idempotency_key="checkpoint")
                assert (await execution.wait(timeout_seconds=10)).result.output == {"text": _OUTPUT}
                row = await _compact_row(state, execution.execution_id)
                refs = _payload_refs(row)
                assert set(refs) == {_INPUT_SLOT, _RESULT_SLOT, _OUTPUT_SLOT}
                stored = await state.execution.executions.get_fresh(execution.execution_id, tenant_id=runtime.tenant_id)
                assert stored.restart_checkpoints and stored.status is ExecutionStatus.SUCCEEDED
                assert bindings and calls == [(_INPUT,)]

    steps = SqliteStepStore(database=step_database)
    run, = await steps.list_runs()
    snapshot = await steps.latest_snapshot(run_id=run.run_id)
    assert snapshot is not None
    transcript = NativeCheckpoint.decode(snapshot).messages
    assert [part.content for message in transcript for part in message.parts
            if isinstance(part, (UserPromptPart, TextPart))] == [_INPUT, _OUTPUT]

    async with Runtime.open(name, storage=storage, features=model_features(models, features), metrics=None) as reopened:
        await _inspect_without_payload_reads(reopened, execution.execution_id, monkeypatch)
        replay = await reopened.agents.get().start(_INPUT, idempotency_key="checkpoint")
        assert replay.execution_id == execution.execution_id
        assert (await replay.wait(timeout_seconds=10)).result.output == {"text": _OUTPUT}
        assert calls == [(_INPUT,)]


@pytest.mark.asyncio
async def test_execution_payload_runtime_incomplete_staging_cannot_use_complete_archive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.persistence import PersistenceDomain
    from linktools.ai.persistence.runs import (
        AgentRunCheckpoint, AgentRunRecord, InMemoryStepArchive, RuntimeAgentRunStore, StagingAgentRunStore,
    )

    staging = StagingAgentRunStore()
    archive = InMemoryStepArchive(PersistenceDomain.CONVERSATION)
    owner = RuntimeAgentRunStore(staging, conversation_archive=archive, execution_archive=None)
    await owner.initialize()
    try:
        run = AgentRunRecord("attempt")
        staging.register_agent_run_local(run)
        staging.save_checkpoint_local(AgentRunCheckpoint("attempt", 1, [], state="interrupted"))
        await archive.materialize_checkpoint(run, AgentRunCheckpoint("attempt", 1, [], state="complete"))
        owner.bind_execution_persistence("attempt", conversation=True, restart_checkpoints=False, durable_trace=False)

        async def reject_fallback(*args, **kwargs):
            raise AssertionError("an existing incomplete staging checkpoint must not use a completed archive")

        monkeypatch.setattr(archive, "latest_checkpoint", reject_fallback)
        with pytest.raises(AIError) as failure:
            await owner.verify_terminal_attempts(candidate_agent_run_ids=("attempt",), required_agent_run_id="attempt")
        assert failure.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await owner.preflight_close()
        await owner.close()


@pytest.mark.asyncio
async def test_execution_payload_runtime_task_producer_keeps_large_json_bodies_compact(tmp_path: Path) -> None:
    from linktools.ai.core import JsonValue, TaskStatus
    from linktools.ai.task import Task, TaskGraph, TaskNode, TaskNodeContext
    from linktools.ai.task.engine import TaskEngine

    input_value = {"body": _INPUT, "options": {"count": 1}}
    output_value = {"body": _OUTPUT, "accepted": True}
    calls = 0

    async def produce(context: TaskNodeContext[None]) -> JsonValue:
        nonlocal calls
        assert context.input == input_value
        calls += 1
        return output_value

    task = Task("execution.payload", produce, effect_policy="none")
    definition = TaskGraph("payload-graph", (TaskNode("produce", task=task, input=input_value),))
    database = tmp_path / "task.db"
    async with Persistence.open("execution-payload-task", storage=PersistenceStorage.sqlite(database)) as state:
        async with TaskEngine.open(task, persistence=state) as engine:
            with _execution_sql_bindings(state) as bindings:
                run = await engine.start(definition, idempotency_key="task-payload")
                assert (await run.wait(timeout_seconds=10)).result.status is TaskStatus.SUCCEEDED
                assert await run.result("produce") == output_value
                execution = await run.execution("produce")
                assert (await execution.wait(timeout_seconds=5)).result.output == output_value
                row = await _compact_row(state, execution.execution_id)
                assert set(_payload_refs(row)) == {_INPUT_SLOT, _RESULT_SLOT, _OUTPUT_SLOT}
                fresh = await state.execution.executions.get_fresh(execution.execution_id, tenant_id=state.tenant_id)
                assert fresh.binding_kind == "task" and fresh.status is ExecutionStatus.SUCCEEDED
                assert fresh.stored_user_input.payload.decode() == input_value
                assert fresh.result.output.kind == "inline" and fresh.result.output.decode() == output_value
                assert bindings and calls == 1

    async with Persistence.open("execution-payload-task", storage=PersistenceStorage.sqlite(database)) as state:
        async with TaskEngine.open(task, persistence=state) as engine:
            restored = await engine.get(definition.graph_id)
            assert await restored.result("produce") == output_value
            fresh = await state.execution.executions.get_fresh(execution.execution_id, tenant_id=state.tenant_id)
            assert fresh.stored_user_input.payload.decode() == input_value
            assert fresh.result.output.decode() == output_value
            assert calls == 1
