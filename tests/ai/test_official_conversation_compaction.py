#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conversation retains its compacted context independently of native recovery."""

import asyncio
from contextlib import AsyncExitStack

import pytest
from pydantic_ai.messages import BinaryContent, ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai_harness.step_persistence import InMemoryStepStore, StepPersistence

from linktools.ai.capability import AgentContext, ExecutionCapability, StepPersistenceCapability
from linktools.ai.core import ExecutionStatus, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistencePlan, PersistenceStorage
from linktools.ai.persistence._archive import PreparedArchive
from linktools.ai.persistence._native_content_index import stage_owner
from linktools.ai.persistence._session_transcript import SESSION_TURN_TABLE, scan_ordinary_message_roots
from linktools.ai.persistence.snapshots import SnapshotLimits
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.runtime._regenerable_turn import _RegenerableTurnRunner
from linktools.ai.storage import InMemoryObjectStore, StorageRoute

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


class _HostAuthorization(TenantAuthorizationPolicy):
    """Exercise the authorized SQL Session path."""


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite-warm", "sqlite-cold", "sqlite-snapshot"))
async def test_native_checkpoint_does_not_replace_conversation_compaction(tmp_path, backend) -> None:
    sql = backend != "memory"
    steps = InMemoryStepStore()
    summaries = []
    provider_views = []
    work_calls = []
    completed = False

    async def work(_context: AgentContext[None], index: int) -> str:
        work_calls.append(index)
        return "old result " + str(index)

    async def respond(messages, _info):
        nonlocal completed
        if "context summarization assistant" in str(messages):
            summaries.append(messages)
            return ModelResponse(parts=[TextPart("trusted-runtime-summary")])
        provider_views.append(messages)
        if not completed and len(work_calls) < 14:
            index = len(work_calls)
            return ModelResponse(parts=[TextPart("old reasoning " * 1500 if index < 3 else "recent"),
                ToolCallPart("work", {"index": index}, tool_call_id=f"call-{index}")])
        completed = True
        return ModelResponse(parts=[TextPart("done")])

    features = RuntimeFeatures("native-conversation-compaction")
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.capability(StepPersistenceCapability() if sql else StepPersistence(store=steps), scope="runtime")
    features.tool(work, effect_policy="replay_safe")
    features.agent("default", model="default", allow_tools=("work",), allow_subagents=())
    model = _UsageFunctionModel(respond, profile={"context_window": 3000})
    model_features(_Models(model), features)
    source = StorageRoute.sqlite(tmp_path / "step.db")
    storage = PersistenceStorage(PersistencePlan(
        conversation=source, execution=source, checkpoint=source,
    )) if sql else PersistenceStorage.in_memory()
    namespace = "native-conversation-compaction"
    async with AsyncExitStack() as stack:
        async def open_runtime(route):
            state = await stack.enter_async_context(Persistence.open(namespace, storage=route))
            service = await stack.enter_async_context(Runtime.open(
                persistence=state, features=features, authorization=_HostAuthorization(),
            ))
            return state, service

        persistence, runtime = await open_runtime(storage)
        session = await runtime.agents.get().create_session("session")

        async def start(prompt, request):
            if sql:
                return await runtime.sessions.start(
                    session.session_id, prompt, agent="default", principal=runtime.default_principal,
                    idempotency_key=request,
                )
            return await session.start(prompt, idempotency_key=request)

        first = await start("work", "first")
        result = await first.wait_final(timeout_seconds=10) if sql else (await first.wait()).result
        assert result.status is ExecutionStatus.SUCCEEDED, result
        assert summaries
        count = len(summaries)
        if sql:
            row = await persistence.session_turns.get_turn(first.key)
            assert "compacted_context" in row["result_json"]
            assert row["result_json"]["compacted_context"] != row["result_json"]["transcript"]
        native_steps = await persistence.create_step_store() if sql else steps
        try:
            first_run = (await native_steps.list_runs())[0]
            native = await native_steps.latest_snapshot(run_id=first_run.run_id)
        finally:
            if sql:
                await native_steps.close()
        assert "trusted-runtime-summary" not in str(native.messages)
        assert "old result 0" in str(native.messages)

        if backend in {"sqlite-cold", "sqlite-snapshot"}:
            await stack.aclose()
            if backend == "sqlite-snapshot":
                archive = InMemoryObjectStore()
                limits = SnapshotLimits(max_entries=10000, max_bytes=16000000)
                async with Persistence.open(namespace, storage=storage, read_only=True) as sealed:
                    reference = await sealed.export_snapshot(object_store=archive, limits=limits)
                root = tmp_path / "restored"
                await Persistence.restore_snapshot(reference, object_store=archive, root=root, limits=limits)
                storage = PersistenceStorage.from_root(root)
            persistence, runtime = await open_runtime(storage)
            turns = persistence.session_turns
            if backend == "sqlite-snapshot":
                moved = (await turns.get_turn(first.key))["result_json"]
                group = persistence._stores[PersistenceDomain.CONVERSATION].storage_group
                owned = [root.to_json() async for root in scan_ordinary_message_roots(group)]
                assert moved["transcript"] in owned and moved["compacted_context"] in owned

        second = await start("continue", "second")
        result = await second.wait_final(timeout_seconds=10) if sql else (await second.wait()).result
        assert result.status is ExecutionStatus.SUCCEEDED, result
        assert "trusted-runtime-summary" in str(provider_views[-1])
        assert len(summaries) == count
        if sql:
            turns = persistence.session_turns
            accepted = await turns.get_turn(second.key)
            assert accepted["request_json"]["compacted_context_turn"] == first.key.turn_id
            assert (accepted["context_from"], accepted["context_through"]) == (1, 1)
            positions = await turns.read_context_range(
                "session", after=0, through=1, compacted_context=first.key.turn_id,
            )
            assert "trusted-runtime-summary" in str(positions[0].compacted_context)
            assert "trusted-runtime-summary" not in str(positions[0].messages)
            runner, _ = runtime.sessions._require_turn_sessions()
            request = {
                "context_from": accepted["context_from"], "context_through": accepted["context_through"],
                "lineage": (), "session_context": "all", "prompt": "continue", "stored": None,
                "compacted_context_turn": accepted["request_json"]["compacted_context_turn"],
            }
            assert await runner._read_admitted_history(second.key, **request) == positions[0].compacted_context
            if backend == "sqlite-cold":
                original = (await turns.get_turn(first.key))["result_json"]
                async with turns._context.engine.begin() as connection:
                    await connection.execute(turns._turns.update().filter_by(
                        **turns._turn_values(first.key),
                    ).values(result_json={**original, "compacted_context": original["transcript"]}))
                with pytest.raises(AIError) as forged:
                    await runner._read_admitted_history(second.key, **request)
                assert forged.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


                # Put a real, structurally valid same-owner root on disk. A
                # typed summary and matching response do not authorize media.
                store, archive_provider, native = turns._transcript_write_binding
                media = ModelRequest(parts=[UserPromptPart([BinaryContent(b"x", media_type="image/png")])])
                archive = await archive_provider()
                prepared = PreparedArchive()
                context = positions[0].compacted_context
                invalid_root = await native.prepare_archived_messages((*context[:-1], media, context[-1]), prepared)

                async def install(group):
                    values = turns._turn_values(first.key)

                    async def write(connection):
                        result = await connection.execute(turns._turns.update().filter_by(**values).values(
                            result_json={**original, "compacted_context": invalid_root.to_json()},
                        ))
                        assert result.rowcount == 1
                        stage_owner(connection, SESSION_TURN_TABLE, turns._scope,
                                    (values["session_key"], values["turn_id"]), invalid_root)

                    await group.transaction(store).run_sql(turns._context, write)
                    await archive.publish(group.transaction(archive.store), prepared)

                await store.storage_group.mutate(tuple(dict.fromkeys((store, archive.store))), install)
                with pytest.raises(AIError) as media_denied:
                    await runner._read_admitted_history(second.key, **request)
                assert media_denied.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_recovered_sql_turn_that_executes_model_retains_new_summary(tmp_path, monkeypatch):
    namespace = "native-compaction-recovery"
    storage = PersistenceStorage.sqlite(tmp_path / "recovery.db")
    provider_calls = []
    summaries = []
    stopped = asyncio.Event()

    async def respond(messages, _info):
        if "context summarization assistant" in str(messages):
            summaries.append(messages)
            return ModelResponse(parts=[TextPart("summary written during recovery")])
        provider_calls.append(messages)
        return ModelResponse(parts=[TextPart("done")])

    features = RuntimeFeatures(namespace)
    features.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    model_features(_Models(_UsageFunctionModel(respond, profile={"context_window": 2400})), features)

    async with Runtime.open(namespace, storage=storage, features=features, authorization=_HostAuthorization()) as runtime:
        principal = runtime.default_principal
        await runtime.sessions.create_turn_session("session", principal=principal)
        for i in range(12):
            first = await runtime.sessions.start("session", f"seed {i}", agent="default", principal=principal,
                                                 idempotency_key=f"seed-{i}")
            assert (await first.wait_final(timeout_seconds=10)).status is ExecutionStatus.SUCCEEDED
        assert not summaries
        settle = _RegenerableTurnRunner.settle

        async def stop_before_model(self, turn, **kwargs):
            stopped.set()
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)

        with monkeypatch.context() as fault:
            fault.setattr(_RegenerableTurnRunner, "settle", stop_before_model)
            second = await runtime.sessions.start("session", "large new question " * 3500, agent="default", principal=principal,
                                                  idempotency_key="second")
            await asyncio.wait_for(stopped.wait(), 5)
        assert _RegenerableTurnRunner.settle is settle and len(provider_calls) == 12

    async with Runtime.open(namespace, storage=storage, features=features, authorization=_HostAuthorization()) as runtime:
        turns = runtime._state.session_turns
        async with turns._context.engine.begin() as connection:
            row = await connection.execute(turns._turns.update().filter_by(
                **turns._turn_values(second.key), generation=1,
            ).values(lease_until=0))
            assert row.rowcount == 1
        assert await runtime.sessions.recover_turn("session", second.key.turn_id, principal=principal)
        result = await turns.get_turn(second.key)
        assert (result["generation"], result["state"]) == (2, "succeeded")
        assert len(provider_calls) == 13 and len(summaries) == 1, (len(provider_calls), len(summaries), len(provider_calls[-1]), len(str(provider_calls[-1])))
        assert "compacted_context" in result["result_json"]
        assert "summary written during recovery" in str(provider_calls[-1])
