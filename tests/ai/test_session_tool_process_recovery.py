#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native SQLite turn recovery across actual process loss after a tool effect."""

import asyncio
import multiprocessing
import os
import traceback
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturn, ToolReturnPart, UserPromptPart

from linktools.ai.capability import AgentContext, ExecutionCapability, SessionCapability, StepPersistenceCapability, TraceCapability
from linktools.ai.core import ExecutionStatus, TenantAuthorizationPolicy, ToolOperationStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.turns import SessionTurnKey
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.runtime._step_store import RuntimeStepStore
from linktools.ai.runtime._tool import RuntimeToolOperationBridge

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


_NAME = "native-session-tool-crash"
_SESSION = "session"
_REQUEST = "turn-1"
_INTERRUPT_MARKER = "linktools.ai.interrupted_tool_receipt"


class _HostAuthorization(TenantAuthorizationPolicy):
    """Supply explicit authorization for the live and recovered SQL hosts."""


def _append(path: Path, value: str) -> None:
    with path.open("a", encoding="utf-8") as log:
        log.write(value + "\n")
        log.flush()
        os.fsync(log.fileno())


def _features(
    root: Path, *, interrupted: bool = False, crash: bool = False, traced: bool = False,
    session_checkpoints: bool = False, wrapped_return: bool = False, missing_receipt: bool = False,
) -> RuntimeFeatures:
    async def effect(context: AgentContext[None]) -> str | ToolReturn:
        _append(root / "effects.log", context.execution_id)
        if crash:
            os._exit(91)
        if wrapped_return:
            return ToolReturn("confirmed effect", content="private fixture supplement", metadata={"proof": "fixture"})
        return "confirmed effect"

    async def model(messages, _info):
        results = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
        if results:
            if interrupted:
                [result] = results
                assert result.outcome == "failed" and "Task interrupted; tool outcome unknown." in str(result.content)
                if result.tool_call_id == "stable-call":
                    _append(root / "provider.log", "interruption")
                    if not missing_receipt:
                        # Even a new call ID must not repeat a known interrupted effect.
                        return ModelResponse(parts=[ToolCallPart("effect", {}, tool_call_id="duplicate-attempt")])
                else:
                    assert result.tool_call_id == "duplicate-attempt"
            else:
                assert [(part.tool_call_id, part.content) for part in results] == [("stable-call", "confirmed effect")]
                if wrapped_return:
                    assert results[0].metadata == {"proof": "fixture"}
                    assert [part.content for part in messages[-1].parts if isinstance(part, UserPromptPart)] == [
                        "private fixture supplement",
                    ]
            _append(root / "provider.log", "continuation")
            return ModelResponse(parts=[TextPart("done")])
        _append(root / "provider.log", "initial")
        return ModelResponse(parts=[ToolCallPart("effect", {}, tool_call_id="stable-call")])

    features = RuntimeFeatures(_NAME)
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    step_policy = SessionCapability(checkpoints=True) if session_checkpoints else StepPersistenceCapability()
    features.capability(step_policy, scope="runtime")
    features.tool(effect, effect_policy="non_replay_safe")
    if traced:
        features.capability(TraceCapability(), scope="runtime")
        features.agent("reviewer", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    features.agent("default", model="default", allow_tools=("effect",), allow_skills=(),
                   allow_subagents=("reviewer",) if traced else ())
    return model_features(_Models(_UsageFunctionModel(model)), features)


async def _send(runtime: Runtime):
    return await runtime.sessions.start(
        _SESSION, "inspect", agent=runtime.agents.get(), principal=runtime.default_principal, idempotency_key=_REQUEST,
    )


def _crash_native_process(root_path: str, phase: str) -> None:
    root = Path(root_path)
    original_complete = RuntimeToolOperationBridge.complete
    original_step = RuntimeStepStore.commit_step
    original_begin = RuntimeToolOperationBridge.begin

    async def begin(bridge, *args, **kwargs):
        if phase == "before_tool_receipt":
            _append(root / "before-tool.log", bridge._execution_id)
            os._exit(91)
        return await original_begin(bridge, *args, **kwargs)

    async def complete(bridge, decision, result):
        completed = await original_complete(bridge, decision, result)
        if phase == "tool_completed":
            os._exit(91)
        return completed

    async def commit_step(store, **kwargs):
        await original_step(store, **kwargs)
        snapshot = kwargs.get("snapshot")
        if (phase in {"tool_checkpoint", "wrapped_tool_checkpoint_trace"}
                and snapshot is not None and snapshot.messages
                and any(isinstance(part, ToolReturnPart) for part in snapshot.messages[-1].parts)):
            # Step has durably retained the tool return. A hard exit here
            # deliberately does not wait for optional Trace publication.
            os._exit(91)

    RuntimeToolOperationBridge.begin = begin
    RuntimeToolOperationBridge.complete = complete
    RuntimeStepStore.commit_step = commit_step

    async def run() -> None:
        async with (
            Persistence.open(_NAME, storage=PersistenceStorage.from_root(root / "runtime")) as state,
            Runtime.open(persistence=state, features=_features(
                root, crash=phase.startswith("tool_interrupted"), traced=phase.endswith("_trace"),
                session_checkpoints=phase.endswith("_session_capability"), wrapped_return=phase == "wrapped_tool_checkpoint_trace",
            ),
                         authorization=_HostAuthorization()) as runtime,
        ):
            await runtime.agents.get().create_session(_SESSION)
            turn = await _send(runtime)
            await turn.wait_final(timeout_seconds=15)
        raise AssertionError("native SQLite crash boundary was not reached")

    asyncio.run(run())


@contextmanager
def _recovery_diagnostic(phase: str):
    try:
        yield
    except AIError as error:
        # CI summarizes failures without a traceback. Report only typed codes,
        # known internal reasons and frame locations, never exception payloads.
        details = []
        cause = error
        for _ in range(3):
            if cause is None:
                break
            code = cause.code.value if isinstance(cause, AIError) else type(cause).__name__
            reason = cause.safe_details.get("reason") if isinstance(cause, AIError) else None
            frames = list(traceback.walk_tb(cause.__traceback__))[-6:]
            sites = ",".join(f"{Path(frame.f_code.co_filename).name}:{lineno}:{frame.f_code.co_name}"
                             for frame, lineno in frames)
            if reason in ("native_subagent_trace_incomplete", "native_child_reservation_missing"):
                code += f":{reason}"
            details.append(f"{code}[{sites}]")
            cause = cause.__cause__
        raise AssertionError(f"{phase}: {' -> '.join(details)}") from None


def _crash_interruption_recovery(root_path: str, turn_id: str, traced: bool, missing_receipt: bool = False) -> None:
    root = Path(root_path)
    commit = RuntimeStepStore.commit_step

    async def die_after_feedback(store, **kwargs):
        try:
            await commit(store, **kwargs)
            snapshot = kwargs.get("snapshot")
            if snapshot is not None:
                retained = await store.latest_snapshot(run_id=turn_id)
                if retained is not None and any(
                    isinstance(part, ToolReturnPart) and isinstance(part.metadata, Mapping)
                    and _INTERRUPT_MARKER in part.metadata
                    for message in retained.messages for part in message.parts
                ):
                    os._exit(91)
        except Exception:
            # The background producer can report a failure before the child
            # process reaches the crash marker. Retain the original traceback.
            _append(root / "interruption-recovery-error.log", traceback.format_exc())
            raise

    RuntimeStepStore.commit_step = die_after_feedback

    async def run():
        async with (
            Persistence.open(_NAME, storage=PersistenceStorage.from_root(root / "runtime")) as state,
            Runtime.open(persistence=state, features=_features(root, interrupted=True, traced=traced, missing_receipt=missing_receipt),
                         authorization=_HostAuthorization()) as runtime,
        ):
            key = SessionTurnKey(_SESSION, turn_id)
            await _expire(state.session_turns, key, generation=1)
            recovered = await runtime.sessions.recover_turn(_SESSION, turn_id, principal=runtime.default_principal)
            assert recovered, "interrupted turn was not recovered"
        raise AssertionError("recovery feedback checkpoint was not reached")

    try:
        asyncio.run(run())
    except Exception:
        _append(root / "interruption-recovery-error.log", traceback.format_exc())
        raise


async def _expire(turns, key: SessionTurnKey, *, generation: int) -> None:
    async with turns._context.engine.begin() as connection:
        expired = await connection.execute(turns._turns.update().filter_by(
            **turns._turn_values(key), generation=generation, state="accepted",
        ).values(lease_until=0))
        assert expired.rowcount == 1


async def _run_until_crash(target, *args):
    process = multiprocessing.get_context("spawn").Process(target=target, args=args)
    process.start()
    try:
        await asyncio.to_thread(process.join, 25)
        if process.exitcode != 91 and target is _crash_interruption_recovery:
            diagnostic = Path(args[0]) / "interruption-recovery-error.log"
            pytest.fail(diagnostic.read_text() if diagnostic.exists() else
                        f"feedback checkpoint process exited without a diagnostic: {process.exitcode}")
        assert process.exitcode == 91, (target, args, process.exitcode)
    finally:
        if process.is_alive():
            process.kill()
            await asyncio.to_thread(process.join, 5)
        process.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", (
    "tool_completed", "tool_checkpoint", "tool_interrupted", "tool_interrupted_twice", "tool_interrupted_twice_trace",
    "tool_interrupted_session_capability", "wrapped_tool_checkpoint_trace",
))
async def test_native_sqlite_tool_turn_recovers_after_process_exit_without_replaying_effect(tmp_path: Path, phase: str):
    await _run_until_crash(_crash_native_process, str(tmp_path), phase)
    interrupted = phase.startswith("tool_interrupted")
    second_crash = phase.startswith("tool_interrupted_twice")
    traced = phase.endswith("_trace")
    effect_log = tmp_path / "effects.log"
    provider_log = tmp_path / "provider.log"
    [turn_id] = effect_log.read_text().splitlines()
    assert provider_log.read_text().splitlines() == ["initial"]
    key = SessionTurnKey(_SESSION, turn_id)
    if second_crash:
        await _run_until_crash(_crash_interruption_recovery, str(tmp_path), turn_id, traced)
    predecessor = 2 if second_crash else 1
    async with (
        Persistence.open(_NAME, storage=PersistenceStorage.from_root(tmp_path / "runtime")) as state,
        Runtime.open(persistence=state, features=_features(
            tmp_path, interrupted=interrupted, traced=traced,
            session_checkpoints=phase.endswith("_session_capability"),
            wrapped_return=phase == "wrapped_tool_checkpoint_trace",
        ),
                     authorization=_HostAuthorization()) as runtime,
    ):
        turns = state.session_turns
        tools = state.checkpoint.tools
        tenant, principal = runtime.tenant_id, runtime.default_principal
        before = await turns.get_turn(key)
        assert before["state"] == "accepted" and before["generation"] == predecessor
        [receipt] = await tools.list_by_execution(turn_id, tenant_id=tenant)
        assert receipt.tool_call_id == "stable-call"
        assert receipt.status is (ToolOperationStatus.CLAIMED if interrupted else ToolOperationStatus.COMPLETED)
        if phase == "wrapped_tool_checkpoint_trace":
            steps = await state.create_step_store()
            try:
                checkpoint = await steps.latest_snapshot(run_id=turn_id)
                assert checkpoint is not None and checkpoint.messages
                parts = checkpoint.messages[-1].parts
                [saved] = [part for part in parts if isinstance(part, ToolReturnPart)]
                assert saved.tool_call_id == "stable-call" and saved.content == "confirmed effect"
                assert saved.metadata == {"proof": "fixture"}
                assert [part.content for part in parts if isinstance(part, UserPromptPart)] == [
                    "private fixture supplement",
                ]
            finally:
                await steps.close()
        repeated = await _send(runtime)
        assert repeated.key == key
        assert await turns.get_turn(key) == before
        assert provider_log.read_text().splitlines() == ["initial"]
        with _recovery_diagnostic("recovery before lease expiry"):
            assert not await runtime.sessions.recover_turn(_SESSION, turn_id, principal=principal)
        # Simulate the DB lease reaching its expiry without sleeping the worker
        # lease duration. Recovery must acquire a new generation itself.
        await _expire(turns, key, generation=predecessor)
        with _recovery_diagnostic("recovery after lease expiry"):
            assert await runtime.sessions.recover_turn(_SESSION, turn_id, principal=principal)
        with _recovery_diagnostic("recovered terminal wait"):
            result = await repeated.wait_final(timeout_seconds=10)
        assert result.key == key and result.status is ExecutionStatus.SUCCEEDED
        assert result.view["output"] == {"text": "done"}
        assert (await turns.get_turn(key))["generation"] == predecessor + 1
        assert (await turns.get_session(_SESSION))["active_turn_id"] is None
        assert await tools.list_by_execution(turn_id, tenant_id=tenant) == (receipt,)
        timeline = (await runtime.history.turn_session(_SESSION, principal=principal))["timeline"]
        [published] = timeline["items"]
        assert published["turn_id"] == turn_id and published["conversation_committed"]
        assert published["user_input"]["prompt"] == "inspect"
        if not interrupted:
            [operation] = published["process"]
            assert operation["key"] == "tool:stable-call" and operation["status"] == "SUCCEEDED"
        assert await state.execution.executions.get(turn_id, tenant_id=tenant) is None
        if traced:
            assert not await state.execution.executions.list_children(turn_id, tenant_id=tenant)
        assert (await _send(runtime)).key == key
    assert effect_log.read_text().splitlines() == [turn_id]
    assert provider_log.read_text().splitlines() == (
        ["initial", "interruption", "continuation"] if interrupted else ["initial", "continuation"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("second_crash", (False, True))
async def test_native_sqlite_recover_missing_tool_receipt_without_dispatch(
    tmp_path: Path, second_crash: bool,
) -> None:
    await _run_until_crash(_crash_native_process, str(tmp_path), "before_tool_receipt")
    turn_id = (tmp_path / "before-tool.log").read_text().strip()
    key = SessionTurnKey(_SESSION, turn_id)
    if second_crash:
        await _run_until_crash(_crash_interruption_recovery, str(tmp_path), turn_id, False, True)
    async with (
        Persistence.open(_NAME, storage=PersistenceStorage.from_root(tmp_path / "runtime")) as state,
        Runtime.open(
            persistence=state, features=_features(tmp_path, interrupted=True, missing_receipt=True),
            authorization=_HostAuthorization(),
        ) as runtime,
    ):
        assert not await state.checkpoint.tools.list_by_execution(turn_id, tenant_id=runtime.tenant_id)
        await _expire(state.session_turns, key, generation=2 if second_crash else 1)
        assert await runtime.sessions.recover_turn(_SESSION, turn_id, principal=runtime.default_principal)
        result = await (await _send(runtime)).wait_final(timeout_seconds=10)
        assert result.status is ExecutionStatus.SUCCEEDED
        assert result.view["output"] == {"text": "done"}
        assert not await state.checkpoint.tools.list_by_execution(turn_id, tenant_id=runtime.tenant_id)
    assert not (tmp_path / "effects.log").exists()
    assert (tmp_path / "provider.log").read_text().splitlines() == ["initial", "interruption", "continuation"]


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_checkpoint", (False, True))
async def test_native_sqlite_unresumable_turn_only_offers_fenced_abandon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing_checkpoint: bool,
) -> None:
    from dataclasses import replace

    await _run_until_crash(_crash_native_process, str(tmp_path), "tool_completed")
    turn_id = (tmp_path / "effects.log").read_text().strip()
    key = SessionTurnKey(_SESSION, turn_id)
    async with (
        Persistence.open(_NAME, storage=PersistenceStorage.from_root(tmp_path / "runtime")) as state,
        Runtime.open(persistence=state, features=_features(tmp_path),
                     authorization=_HostAuthorization()) as runtime,
    ):
        repository = state.checkpoint.tools
        original = type(repository).list_by_agent_run

        async def mismatched(repo, *args, **kwargs):
            values = await original(repo, *args, **kwargs)
            return tuple(replace(value, tool_name="other_tool") for value in values)

        if missing_checkpoint:
            steps = await state.create_step_store()
            try:
                async def no_checkpoint(*args, **kwargs):
                    return None

                monkeypatch.setattr(type(steps), "capture_turn_checkpoint", no_checkpoint)
            finally:
                await steps.close()
            stage, reason, generation = "takeover", "checkpoint_missing_before_takeover_commit", 1
        else:
            monkeypatch.setattr(type(repository), "list_by_agent_run", mismatched)
            stage, reason, generation = "tool_receipts", "pending_tool_receipt_name_mismatch", 2
        await _expire(state.session_turns, key, generation=1)
        with pytest.raises(AIError) as failure:
            await runtime.sessions.recover_turn(_SESSION, turn_id, principal=runtime.default_principal)
        assert failure.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
        assert failure.value.safe_details == {
            "recovery_stage": stage, "reason": reason, "abandonable": True, "generation": generation,
        }
        row = await state.session_turns.get_turn(key)
        assert (row["state"], row["generation"]) == ("accepted", generation)
        assert not await runtime.sessions.abandon_turn(
            _SESSION, turn_id, expected_generation=3 - generation, principal=runtime.default_principal,
        )
        assert await runtime.sessions.abandon_turn(
            _SESSION, turn_id, expected_generation=generation, principal=runtime.default_principal,
        )
        assert (await state.session_turns.get_turn(key))["state"] == "failed"
        assert (await state.session_turns.get_session(_SESSION))["active_turn_id"] is None
    assert (tmp_path / "effects.log").read_text().splitlines() == [turn_id]
    assert (tmp_path / "provider.log").read_text().splitlines() == ["initial"]
