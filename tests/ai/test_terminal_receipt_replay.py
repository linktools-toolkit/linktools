#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retained terminal retries confirm canonical results without replaying facts."""

from dataclasses import fields, replace
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.capability import ExecutionCapability, TraceCapability
from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ExecutionTerminalCommit, ExecutionTerminalCommitResult
from linktools.ai.persistence import Persistence, PersistencePlan, PersistenceStorage
from linktools.ai.persistence._runtime_commands import StateCommands
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import StorageRoute

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


@pytest.mark.asyncio
@pytest.mark.parametrize("split_release_ack", (False, True))
async def test_retained_terminal_replay_confirms_canonical_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, split_release_ack: bool,
) -> None:
    async def respond(messages, info):
        return ModelResponse(parts=[TextPart("retained answer")])

    features = RuntimeFeatures("terminal-receipt-replay")
    features.agent("default", model="default", allow_tools=())
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.capability(TraceCapability(), scope="runtime")
    execution_route = StorageRoute.sqlite(tmp_path / "execution.db")
    storage = (PersistenceStorage.from_plan(PersistencePlan(
        execution=execution_route,
        trace=execution_route,
        conversation=StorageRoute.sqlite(tmp_path / "conversation.db"),
    )) if split_release_ack else PersistenceStorage.sqlite(tmp_path / "state.db"))
    original = StateCommands.commit_terminal_checkpoint
    confirmed: list[ExecutionTerminalCommitResult] = []
    errors: list[BaseException] = []
    lost_acknowledgements = 0

    async with Persistence.open("terminal-receipt-replay", storage=storage) as state, Runtime.attach(
        persistence=state, features=model_features(_Models(_UsageFunctionModel(respond)), features),
    ) as runtime:
        sessions = state.conversation.sessions
        events = state.execution.events
        tenant_id = state.tenant_id

        async def receipt(execution_id: str):
            return (
                await events.list(execution_id, tenant_id=tenant_id, after_event_seq=0, limit=100),
                await sessions.list_timeline_commits(
                    "session", tenant_id=tenant_id, start_turn_seq=1, end_turn_seq=2,
                ),
                await sessions.timeline_head("session", tenant_id=tenant_id),
            )

        async def replay(
            commands: StateCommands, commit: ExecutionTerminalCommit, **kwargs,
        ) -> ExecutionTerminalCommitResult:
            nonlocal lost_acknowledgements
            first = await original(commands, commit, **kwargs)
            if kwargs.get("session_id") != "session":
                return first
            try:
                # Replay the unchanged public Runtime command before local
                # handoff advances its independent control metadata.
                assert commit.execution.completed_at is None
                assert commit.execution.result is None
                assert first.execution.completed_at is not None
                assert first.execution.completed_at == commit.execution.updated_at
                assert first.execution.result == commit.result
                before = await receipt(commit.execution.execution_id)
                assert len(before[1]) == before[2] == 1
                mutate = sessions.state_store.mutate

                async def lose_release_reply(callback, *args, **options):
                    nonlocal lost_acknowledgements
                    value = await mutate(callback, *args, **options)
                    if callback.__name__ == "release_conversation" and not lost_acknowledgements:
                        lost_acknowledgements += 1
                        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
                    return value

                with monkeypatch.context() as patch:
                    if split_release_ack:
                        patch.setattr(sessions.state_store, "mutate", lose_release_reply)
                    second = await original(commands, commit, **kwargs)
                assert second.execution == first.execution
                assert second.result == first.result
                assert await receipt(commit.execution.execution_id) == before

                damaged = replace(commit, result=replace(
                    commit.result, created_at=commit.result.created_at + timedelta(seconds=1),
                ))
                with pytest.raises(AIError) as rejected:
                    await original(commands, damaged, **kwargs)
                assert rejected.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
                assert await receipt(commit.execution.execution_id) == before
                assert await state.execution.executions.get(
                    commit.execution.execution_id, tenant_id=tenant_id,
                ) == first.execution
                if not split_release_ack:
                    session_record = await sessions.get("session", tenant_id=tenant_id)
                    cursor = session_record.continuation
                    assert cursor.checkpoint_sequence is not None and cursor.projection_digest is not None
                    supplied = kwargs["next_cursor"]
                    for damaged_cursor in (
                        replace(supplied, checkpoint_sequence=cursor.checkpoint_sequence + 1,
                                projection_digest=cursor.projection_digest),
                        replace(supplied, projection_digest="0" * 64),
                    ):
                        with pytest.raises(AIError) as invalid_cursor:
                            await original(commands, commit, **{**kwargs, "next_cursor": damaged_cursor})
                        assert invalid_cursor.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
                    checkpoint = kwargs["conversation_checkpoint"]
                    with pytest.raises(AIError) as invalid_checkpoint:
                        await original(commands, commit, **{**kwargs, "conversation_checkpoint": replace(
                            checkpoint, timestamp=checkpoint.timestamp + timedelta(seconds=1),
                        )})
                    assert invalid_checkpoint.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
                    assert await receipt(commit.execution.execution_id) == before
                    assert await sessions.get("session", tenant_id=tenant_id) == session_record
                confirmed.append(second)
            except BaseException as error:
                expected = replace(
                    commit.execution, revision=first.execution.revision,
                    event_seq=first.execution.event_seq, result=commit.result,
                    history_seal_digest=first.execution.history_seal_digest,
                ).finalize_terminal(commit.result)
                error.add_note(str({
                    field.name: (getattr(first.execution, field.name), getattr(expected, field.name))
                    for field in fields(expected)
                    if getattr(first.execution, field.name) != getattr(expected, field.name)
                }))
                errors.append(error)
                raise
            return first

        monkeypatch.setattr(StateCommands, "commit_terminal_checkpoint", replay)
        session = await runtime.agents.get().create_session("session")
        outcome = (await session.run("original input", timeout_seconds=10)).result
        if errors:
            raise errors[0]
        assert outcome.status is ExecutionStatus.SUCCEEDED
        assert len(confirmed) == 1
        assert lost_acknowledgements == int(split_release_ack)
        timeline = (await session.timeline()).items
        assert len(timeline) == 1 and timeline[0].conversation_committed
        assert timeline[0].output == outcome.output
