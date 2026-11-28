#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Current SQLite history survives reopening and snapshot restoration."""

import json
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, UserPromptPart
from pydantic_ai.models.function import AgentInfo

from linktools.ai.core import CommitPolicy, ExecutionStatus, Principal, TenantAuthorizationPolicy
from linktools.ai.capability import TraceCapability
from linktools.ai.history import History
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures

from . import _runtime_test_helpers as helpers


class _HostAuthorization(TenantAuthorizationPolicy):
    """Authorize this test's explicitly trusted local SQLite owner."""


def _features(*, trace: bool = False) -> RuntimeFeatures:
    application = RuntimeFeatures("application", defaults={"execution": {"retain_history": True}})
    if trace:
        application.capability(TraceCapability(commit=CommitPolicy.sync()), scope="runtime")
    application.agent("default", model="default", allow_tools=())
    return helpers.model_features(helpers.RuntimeUsageModels(), application)


async def _seed_current_history(path: Path) -> tuple[Principal, str]:
    async with (
        Persistence.open("sqlite-history", storage=PersistenceStorage.sqlite(path)) as state,
        Runtime.open(persistence=state, features=_features(trace=True), authorization=_HostAuthorization()) as runtime,
    ):
        session = await runtime.agents.get("default").create_session("session")
        result = await _run(runtime, "retained old question", "original")
        assert result.view == (await session.timeline())["items"][0]
        await helpers._wait_for_committed(
            lambda: runtime.history.turn_model_interactions("session", result.key.turn_id, principal=runtime.default_principal),
            lambda page: len(page.items) == 1,
        )
        return runtime.default_principal, result.key.turn_id


async def _run(runtime: Runtime, prompt: str, request_key: str):
    turn = await runtime.sessions.start(
        "session", prompt, agent="default", principal=runtime.default_principal, idempotency_key=request_key,
    )
    result = await turn.wait_final(timeout_seconds=10)
    assert result.status is ExecutionStatus.SUCCEEDED
    return result


async def _turn_evidence(history: History, turn_id: str, principal: Principal) -> dict[str, object]:
    interactions = await history.turn_model_interactions(
        "session", turn_id, principal=principal, local_workspace=True, include_content=True,
    )
    interaction, = interactions.items
    assert interaction["status"] == "SUCCEEDED"
    assert interaction["model_request_seq"] == 1
    assert (interaction["agent_run_seq"], interaction["step_index"], interaction["purpose"]) == (1, 1, "agent")
    metadata = await history.turn_model_interactions("session", turn_id, principal=principal)
    metadata_item, = metadata.items
    assert metadata_item["request"] == {} and metadata_item["response"] is None
    assert metadata_item["usage"] == interaction["usage"]
    assert interaction["request"]["messages"]
    assert "retained old question" in json.dumps(interaction["request"])
    assert "done" in json.dumps(interaction["response"])
    usage = interaction["usage"]
    assert (usage["input_tokens"], usage["output_tokens"]) == (101, 202)
    assert (usage["cache_read_tokens"], usage["cache_write_tokens"]) == (303, 404)
    trace = await history.turn_trace("session", turn_id, principal=principal)
    assert [item.payload["kind"] for item in trace.items] == ["MODEL_REQUEST", "MODEL_RESPONSE"]
    assert [item.payload["status"] for item in trace.items] == ["STARTED", "SUCCEEDED"]
    assert all(item.turn_id == turn_id for item in trace.items)
    result = await history.turn_result("session", turn_id, principal=principal)
    assert result["conversation_committed"]
    assert result["user_input"]["prompt"] == "retained old question"
    assert result["output"] == {"text": "done"}
    assert [(item["item_kind"], item["content"]) for item in result["items"]] == [("assistant", "done")]
    return {"interactions": interactions.items, "trace": trace.items, "turn": result}


@pytest.mark.asyncio
async def test_sqlite_history_retains_prior_turn_when_resumed_and_reopened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "history.db"
    principal, turn_id = await _seed_current_history(path)
    observed: list[list[str]] = []
    original_model = helpers._runtime_usage_model

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        observed.append([
            part.content for message in messages for part in message.parts
            if isinstance(part, UserPromptPart) and isinstance(part.content, str)
        ])
        return await original_model(messages, info)

    monkeypatch.setattr(helpers, "_runtime_usage_model", model)
    async with (
        Persistence.open("sqlite-history", storage=PersistenceStorage.sqlite(path)) as state,
        Runtime.open(persistence=state, features=_features(), authorization=_HostAuthorization()) as runtime,
    ):
        session = await runtime.sessions.get("session", principal=principal)
        evidence = await _turn_evidence(runtime.history, turn_id, principal)
        old = await session.timeline()
        assert old["items"] == [evidence["turn"]]
        result = await _run(runtime, "new question", "next")
        replay = await _run(runtime, "new question", "next")
        assert replay.key == result.key and replay.view == result.view
        assert observed == [["retained old question", "new question"]]
        appended = await session.timeline()
        assert appended["items"] == [*old["items"], result.view]
        assert result.view["user_input"]["prompt"] == "new question"
        assert result.view["output"] == {"text": "done"}
        assert await _turn_evidence(runtime.history, turn_id, principal) == evidence

    async with (
        Persistence.open("sqlite-history", storage=PersistenceStorage.sqlite(path), read_only=True) as state,
        History.open(persistence=state, authorization=_HostAuthorization()) as history,
    ):
        restored = (await history.turn_session("session", principal=principal))["timeline"]
        assert restored == appended
        assert await _turn_evidence(history, turn_id, principal) == evidence


@pytest.mark.asyncio
async def test_sqlite_explicit_checkpoint_boundary_excludes_later_observation(tmp_path: Path) -> None:
    from pydantic_ai.messages import TextPart

    from linktools.ai.persistence._step_contracts import AgentRunCheckpoint

    from .test_step_archive_read_boundaries import _archive

    async with _archive(tmp_path, "sqlite") as (_state, archive, run):
        await archive.register_agent_run(run, execution_id="execution")
        first = ModelResponse(parts=[TextPart("checkpoint message")])
        later = ModelResponse(parts=[TextPart("later observation")])
        prepared = await archive.prepare_checkpoints(
            run, (AgentRunCheckpoint(
                "run", 1, [first], transcript_message_count_before=0,
            ),),
        )
        await archive.sync_prepared_projection(
            run, events=(), checkpoints=prepared.checkpoints,
            execution_id="execution", producer_generation=1,
        )
        observation = await archive.transcript_repository.prepare_observation(
            "run", (later,), first_message_index=1, pending=None, pending_keys=(),
        )
        await archive.sync_prepared_projection(
            run, events=(), checkpoints=(), observation=observation,
            execution_id="execution", producer_generation=1,
        )
        assert [message async for message in archive.iter_messages(agent_run_id="run")] == [first, later]
        checkpoint = await archive.latest_checkpoint(agent_run_id="run")
        assert checkpoint is not None
        assert checkpoint.messages == [first]


@pytest.mark.asyncio
async def test_sqlite_history_survives_snapshot_restore(tmp_path: Path) -> None:
    from linktools.ai.persistence.snapshots import SnapshotLimits
    from linktools.ai.storage import InMemoryObjectStore

    path = tmp_path / "history.db"
    principal, turn_id = await _seed_current_history(path)
    objects = InMemoryObjectStore("snapshot")
    limits = SnapshotLimits(max_entries=1000, max_bytes=1024 * 1024)
    async with Persistence.open("sqlite-history", storage=PersistenceStorage.sqlite(path), read_only=True) as state:
        async with History.open(persistence=state, authorization=_HostAuthorization()) as history:
            timeline = (await history.turn_session("session", principal=principal))["timeline"]
            evidence = await _turn_evidence(history, turn_id, principal)
            assert timeline["items"] == [evidence["turn"]]
        snapshot = await state.export_snapshot(object_store=objects, limits=limits)
    restored_root = tmp_path / "restored"
    await Persistence.restore_snapshot(
        snapshot, object_store=objects, root=restored_root, limits=limits,
    )
    async with (
        Persistence.open("sqlite-history", storage=PersistenceStorage.from_root(restored_root)) as state,
        Runtime.open(persistence=state, features=_features(), authorization=_HostAuthorization()) as runtime,
    ):
        assert await _turn_evidence(runtime.history, turn_id, principal) == evidence
        session = await runtime.sessions.get("session", principal=principal)
        assert await session.timeline() == timeline
