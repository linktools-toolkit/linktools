#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Runtime.resume -- the pause/approve/resume round trip (scenario).

Runtime.resume(run_id, spec) loads a paused run, deserializes its checkpoint's
message history, transitions WAITING_APPROVAL -> RUNNING, and re-enters
AgentEngine.run_stream with message_history=<deserialized messages>. The
GovernedToolInvoker's resume gate (_already_approved) recognizes the now-APPROVED
request and lets the tool execute instead of re-raising RunPaused.

The full round trip at the Runtime level:

  1. ``runtime.run_stream(spec, "call risky", ...)`` -- the model emits a
     ToolCallPart for "risky"; the executor (pause_on_approval=True) raises
     RunPaused; run_stream checkpoints the partial message history and
     yields ``{"type": "paused", ...}``.
  2. ``storage.approvals.approve(approval_id, ...)`` -- the human approves.
  3. ``runtime.resume(run_id, spec)`` -- deserializes the checkpoint,
     transitions WAITING_APPROVAL -> RUNNING, re-enters run_stream with
     message_history. The graph resumes: the pending tool call executes
     (resume gate lets it through), the model emits "done", and the run
     SUCCEEDS.

All three phases run inside one ``asyncio.run`` so the FilesystemApprovalStore's
``asyncio.Lock`` stays bound to one event loop (the lock is acquired during
both the pause create and the approve resolve)."""

import asyncio
import json
from datetime import datetime, timezone

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from linktools.ai.agent.spec import AgentSpec, PromptSpec, ToolRef
from linktools.ai.capability.models import CapabilityBundle
from linktools.ai.capability.provider import CapabilityProvider
from linktools.ai.tool.models import (
    ManagedToolDefinition,
    ToolContribution,
    ToolDescriptor,
)
from linktools.ai.runtime import RuntimeDependencies
from linktools.ai.model.registry import ModelRegistry
from linktools.ai.errors import InvalidRunTransitionError, RunNotFoundError
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.router import ModelRouter
from linktools.ai.governance.policy.approval import ApprovalRule
from linktools.ai.governance.policy.engine import PolicyEngine
from linktools.ai.run.models import RunInput, RunnableType, RunRecord, RunStatus
from linktools.ai.runtime import Runtime
from linktools.ai.session.models import SessionRecord, SessionStatus
from linktools.ai.storage.facade import FilesystemStorage
from linktools.ai.tool.executor import GovernedToolInvoker

TOOL_NAME = "risky"


class _RiskyProvider(CapabilityProvider):
    supported_kinds = ("test",)

    async def resolve(self, ref, context):
        async def risky(x: int) -> int:
            return x * 2

        return CapabilityBundle(
            tool_contributions=(
                ToolContribution(
                    tools=(
                        ManagedToolDefinition(
                            descriptor=ToolDescriptor(
                                name=TOOL_NAME,
                                source="test",
                                category="discovery",
                                risk="high",
                                mutating=False,
                            ),
                            handler=risky,
                        ),
                    )
                ),
            )
        )


# -- Model fixtures ---------------------------------------------------------


def _model_fn(messages, info: AgentInfo) -> ModelResponse:
    """Turn 1: emit a ToolCallPart for the risky tool. Turn 2 (after the tool
    has returned): emit TextPart("done"). The turn is determined by whether
    the history already contains a ToolReturnPart for risky."""
    for m in messages:
        parts = getattr(m, "parts", None) or []
        for p in parts:
            if (
                isinstance(p, ToolReturnPart)
                and getattr(p, "tool_name", None) == TOOL_NAME
            ):
                return ModelResponse(parts=[TextPart(content="done")])
    return ModelResponse(parts=[ToolCallPart(tool_name=TOOL_NAME, args={"x": 1})])


async def _stream_fn(messages, info: AgentInfo):
    for m in messages:
        parts = getattr(m, "parts", None) or []
        for p in parts:
            if (
                isinstance(p, ToolReturnPart)
                and getattr(p, "tool_name", None) == TOOL_NAME
            ):
                yield "done"
                return
    yield {0: DeltaToolCall(name=TOOL_NAME, json_args=json.dumps({"x": 1}))}


def _registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register(
        "test-model",
        model=FunctionModel(_model_fn, stream_function=_stream_fn),
    )
    return registry


def _spec() -> AgentSpec:
    return AgentSpec(
        id="agent-resume",
        name="resume-agent",
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="hi"),
        output_schema=str,
        tools=(ToolRef(kind="test", name=TOOL_NAME),),
    )


def _build_runtime(tmp_path) -> "tuple[Runtime, FilesystemStorage]":
    storage = FilesystemStorage(root=tmp_path)
    executor = GovernedToolInvoker(
        policy=PolicyEngine(rules=(ApprovalRule(require_for=frozenset({TOOL_NAME})),)),
        approval_store=storage.approvals,
    )
    runtime = Runtime.build(
        storage=storage,
        model_router=ModelRouter(registry=_registry()),
        tool_executor=executor,
        providers=RuntimeDependencies(capabilities=(_RiskyProvider(),)),
        local_trusted_mode=True,
    )
    return runtime, storage


async def _collect(gen) -> "list[dict]":
    out: "list[dict]" = []
    async for event in gen:
        out.append(event)
    return out


# -- Tests ------------------------------------------------------------------


def test_resume_round_trip_pause_approve_resume_succeeds(tmp_path):
    """Full pause -> approve -> resume -> SUCCEEDED round trip at the Runtime
    level. All three phases run inside one event loop so the
    FilesystemApprovalStore's asyncio.Lock stays bound to one loop."""

    async def _drive():
        runtime, storage = _build_runtime(tmp_path)
        spec = _spec()

        # Create a session.
        now = datetime.now(timezone.utc)
        await storage.sessions.create(
            SessionRecord(
                id="session-r1",
                parent_id=None,
                status=SessionStatus.ACTIVE,
                version=1,
                created_at=now,
                updated_at=now,
            )
        )

        # run_stream pauses on the risky tool.
        pause_events = await _collect(
            runtime.run_stream(
                spec,
                "call risky",
                session_id="session-r1",
                run_id="run-r1",
            )
        )
        paused = next(e for e in pause_events if e["type"] == "paused")
        approval_id = paused["approval_id"]
        assert paused["run_id"] == "run-r1"

        # Run is WAITING_APPROVAL (NOT FAILED).
        record = await storage.runs.get("run-r1")
        assert record.status is RunStatus.WAITING_APPROVAL

        # human approves the pending request.
        await storage.approvals.approve(
            approval_id,
            expected_version=1,
            resolved_by="test",
        )

        # resume re-enters run_stream with message_history.
        resume_events = await _collect(runtime.resume("run-r1"))

        # Final record: SUCCEEDED.
        final = await storage.runs.get("run-r1")
        return resume_events, final

    resume_events, final_record = asyncio.run(_drive())

    # "resumed" signal is yielded first.
    assert resume_events[0]["type"] == "resumed"
    assert resume_events[0]["run_id"] == "run-r1"

    # Tool executed (end event with ok=True).
    tool_ends = [
        e for e in resume_events if e.get("type") == "tool" and e.get("phase") == "end"
    ]
    assert any(e["name"] == TOOL_NAME and e["ok"] is True for e in tool_ends), (
        f"expected tool end event for {TOOL_NAME} with ok=True, got {resume_events}"
    )

    # Text event present.
    text_events = [e for e in resume_events if e.get("type") == "text"]
    assert any("done" in e["text"] for e in text_events), (
        f"expected text event 'done', got {resume_events}"
    )

    # Run transitioned to SUCCEEDED.
    assert final_record.status is RunStatus.SUCCEEDED


def test_resume_run_not_found_raises(tmp_path):
    """Resume on a non-existent run raises RunNotFoundError."""
    runtime, _ = _build_runtime(tmp_path)

    async def _drive():
        try:
            async for _ in runtime.resume("nonexistent"):
                pass
            return None
        except RunNotFoundError:
            return True

    assert asyncio.run(_drive()) is True


def test_resume_not_waiting_approval_raises(tmp_path):
    """Resume on a run that is SUCCEEDED (not WAITING_APPROVAL) raises
    InvalidRunTransitionError."""
    runtime, storage = _build_runtime(tmp_path)

    async def _seed_and_resume():
        now = datetime.now(timezone.utc)
        await storage.runs.create(
            RunRecord(
                id="run-done",
                root_run_id="run-done",
                parent_run_id=None,
                session_id="session-x",
                runnable_id="agent-resume",
                runnable_type=RunnableType.AGENT,
                status=RunStatus.SUCCEEDED,
                input=RunInput(prompt=""),
                result=None,
                error=None,
                version=1,
                created_at=now,
                started_at=now,
                finished_at=now,
            )
        )
        try:
            async for _ in runtime.resume("run-done"):
                pass
            return None
        except InvalidRunTransitionError:
            return True

    assert asyncio.run(_seed_and_resume()) is True


def test_resume_no_checkpoint_raises(tmp_path):
    """Resume on a WAITING_APPROVAL run without a checkpoint raises
    RunNotFoundError."""
    runtime, storage = _build_runtime(tmp_path)

    async def _seed_and_resume():
        now = datetime.now(timezone.utc)
        await storage.runs.create(
            RunRecord(
                id="run-paused-nockpt",
                root_run_id="run-paused-nockpt",
                parent_run_id=None,
                session_id="session-y",
                runnable_id="agent-resume",
                runnable_type=RunnableType.AGENT,
                status=RunStatus.WAITING_APPROVAL,
                input=RunInput(prompt=""),
                result=None,
                error=None,
                version=1,
                created_at=now,
                started_at=now,
                finished_at=None,
            )
        )
        try:
            async for _ in runtime.resume("run-paused-nockpt"):
                pass
            return None
        except RunNotFoundError:
            return True

    assert asyncio.run(_seed_and_resume()) is True


def test_resume_refused_when_snapshot_marked_non_resumable(tmp_path):
    """A run whose persisted snapshot is marked NON_RESUMABLE (§13.7) is refused
    at resume entry -- the guard fires right after the snapshot is read, before
    the checkpoint/approval checks."""
    from linktools.ai.errors import RunNotResumableError
    from linktools.ai.run.definition import (
        RunDefinitionSnapshot,
        serialize_agent_spec,
        spec_fingerprint,
    )

    runtime, storage = _build_runtime(tmp_path)

    async def _seed_and_resume():
        now = datetime.now(timezone.utc)
        await storage.runs.create(
            RunRecord(
                id="run-nonresumable",
                root_run_id="run-nonresumable",
                parent_run_id=None,
                session_id="session-nr",
                runnable_id="agent-resume",
                runnable_type=RunnableType.AGENT,
                status=RunStatus.WAITING_APPROVAL,
                input=RunInput(prompt=""),
                result=None,
                error=None,
                version=1,
                created_at=now,
                started_at=now,
                finished_at=None,
            )
        )
        spec = _spec()
        await storage.run_definitions.create(
            RunDefinitionSnapshot(
                run_id="run-nonresumable",
                runnable_type="agent",
                runnable_id="agent-resume",
                serialized_spec=serialize_agent_spec(spec),
                spec_fingerprint=spec_fingerprint(spec),
                user_id=None,
                tenant_id=None,
                workspace=None,
                provider_revision=None,
                created_at=now,
                manifest={},
                resumability="non_resumable",
            )
        )
        try:
            async for _ in runtime.resume("run-nonresumable"):
                pass
            return None
        except RunNotResumableError:
            return True

    assert asyncio.run(_seed_and_resume()) is True


def test_resume_refused_when_provider_revision_drifted(tmp_path):
    """§13.6: if the resolved model provider's revision changed between prepare
    and resume, resume refuses (ManifestDriftError) instead of silently
    re-resolving against the drifted environment. The snapshot is seeded with a
    manifest that pins a STALE provider revision; the runtime's current
    test-model resolves to a different revision."""
    from linktools.ai.errors import ManifestDriftError
    from linktools.ai.run.definition import (
        RunDefinitionSnapshot,
        serialize_agent_spec,
        spec_fingerprint,
    )
    from linktools.ai.run.manifest import (
        build_execution_manifest,
        manifest_to_dict,
    )

    runtime, storage = _build_runtime(tmp_path)

    async def _seed_and_resume():
        now = datetime.now(timezone.utc)
        await storage.runs.create(
            RunRecord(
                id="run-drift",
                root_run_id="run-drift",
                parent_run_id=None,
                session_id="session-drift",
                runnable_id="agent-resume",
                runnable_type=RunnableType.AGENT,
                status=RunStatus.WAITING_APPROVAL,
                input=RunInput(prompt=""),
                result=None,
                error=None,
                version=1,
                created_at=now,
                started_at=now,
                finished_at=None,
            )
        )
        spec = _spec()

        class _StaleProvider:
            # A revision that cannot match the real test-model's config-hash.
            revision = "stale-revision-that-does-not-match-anything"

        stale_manifest = manifest_to_dict(
            build_execution_manifest(
                spec,
                runnable_type="agent",
                runnable_fingerprint=spec_fingerprint(spec),
                model_provider=_StaleProvider(),
            )
        )
        await storage.run_definitions.create(
            RunDefinitionSnapshot(
                run_id="run-drift",
                runnable_type="agent",
                runnable_id="agent-resume",
                serialized_spec=serialize_agent_spec(spec),
                spec_fingerprint=spec_fingerprint(spec),
                user_id=None,
                tenant_id=None,
                workspace=None,
                provider_revision=None,
                created_at=now,
                manifest=stale_manifest,
                resumability="resumable",
            )
        )
        try:
            async for _ in runtime.resume("run-drift"):
                pass
            return None
        except ManifestDriftError:
            return True

    assert asyncio.run(_seed_and_resume()) is True


def test_resume_refused_when_approval_still_pending(tmp_path):
    """WP-08 §12.6: a run whose approval is still PENDING must not resume -- the
    run stays WAITING_APPROVAL (resume is fail-closed until explicitly approved)."""

    async def _drive():
        runtime, storage = _build_runtime(tmp_path)
        spec = _spec()
        now = datetime.now(timezone.utc)
        await storage.sessions.create(
            SessionRecord(
                id="session-pending",
                parent_id=None,
                status=SessionStatus.ACTIVE,
                version=1,
                created_at=now,
                updated_at=now,
            )
        )
        pause_events = await _collect(
            runtime.run_stream(
                spec,
                "call risky",
                session_id="session-pending",
                run_id="run-pending",
            )
        )
        assert any(e["type"] == "paused" for e in pause_events)
        # Do NOT approve -- the request is still PENDING.
        try:
            async for _ in runtime.resume("run-pending"):
                pass
            return None
        except InvalidRunTransitionError:
            # Run must still be WAITING_APPROVAL (no CAS happened).
            record = await storage.runs.get("run-pending")
            return record.status is RunStatus.WAITING_APPROVAL

    assert asyncio.run(_drive()) is True


def test_resume_refused_when_approval_rejected(tmp_path):
    """WP-08 §12.6: a REJECTED approval must not resume -- fail-closed, run stays
    WAITING_APPROVAL."""

    async def _drive():
        runtime, storage = _build_runtime(tmp_path)
        spec = _spec()
        now = datetime.now(timezone.utc)
        await storage.sessions.create(
            SessionRecord(
                id="session-rej",
                parent_id=None,
                status=SessionStatus.ACTIVE,
                version=1,
                created_at=now,
                updated_at=now,
            )
        )
        pause_events = await _collect(
            runtime.run_stream(
                spec,
                "call risky",
                session_id="session-rej",
                run_id="run-rej",
            )
        )
        paused = next(e for e in pause_events if e["type"] == "paused")
        await storage.approvals.reject(
            paused["approval_id"], expected_version=1, resolved_by="test"
        )
        try:
            async for _ in runtime.resume("run-rej"):
                pass
            return None
        except InvalidRunTransitionError:
            record = await storage.runs.get("run-rej")
            return record.status is RunStatus.WAITING_APPROVAL

    assert asyncio.run(_drive()) is True


def test_resume_persists_non_empty_user_message(tmp_path):
    """WP-11 §15.2: a resumed run's complete commit must persist the ORIGINAL
    user message (carried through record.input.prompt), not an empty string."""

    async def _drive():
        runtime, storage = _build_runtime(tmp_path)
        spec = _spec()
        now = datetime.now(timezone.utc)
        await storage.sessions.create(
            SessionRecord(
                id="session-usr",
                parent_id=None,
                status=SessionStatus.ACTIVE,
                version=1,
                created_at=now,
                updated_at=now,
            )
        )
        pause_events = await _collect(
            runtime.run_stream(
                spec,
                "call risky",
                session_id="session-usr",
                run_id="run-usr",
            )
        )
        paused = next(e for e in pause_events if e["type"] == "paused")
        await storage.approvals.approve(
            paused["approval_id"], expected_version=1, resolved_by="test"
        )
        await _collect(runtime.resume("run-usr"))
        messages = await storage.sessions.list_messages("session-usr")
        return [m for m in messages if m.role.value == "user"]

    user_msgs = asyncio.run(_drive())
    assert len(user_msgs) == 1, f"expected 1 USER message, got {len(user_msgs)}"
    assert user_msgs[0].content == "call risky", (
        f"expected original prompt 'call risky', got {user_msgs[0].content!r}"
    )


class _PromptedRiskyProvider(_RiskyProvider):
    """Same risky tool as _RiskyProvider, but ALSO injects a capability prompt
    section -- reproduces B-02 (resume with capability_prompt non-empty used to
    do ``str + None`` and crash)."""

    async def resolve(self, ref, context):
        bundle = await super().resolve(ref, context)
        return bundle.__class__(
            tool_contributions=bundle.tool_contributions,
            prompt_sections={"catalog": "## Tool Catalog\n- risky: doubles a number"},
        )


def test_resume_with_capability_prompt_does_not_crash(tmp_path):
    """B-02: a capability-enabled agent (with a prompt section) that pauses on
    approval and then resumes must not TypeError on ``capability_prompt + None``."""

    async def _drive():
        storage = FilesystemStorage(root=tmp_path)
        executor = GovernedToolInvoker(
            policy=PolicyEngine(
                rules=(ApprovalRule(require_for=frozenset({TOOL_NAME})),)
            ),
            approval_store=storage.approvals,
        )
        runtime = Runtime.build(
            storage=storage,
            model_router=ModelRouter(registry=_registry()),
            tool_executor=executor,
            providers=RuntimeDependencies(capabilities=(_PromptedRiskyProvider(),)),
            local_trusted_mode=True,
        )
        spec = _spec()
        now = datetime.now(timezone.utc)
        await storage.sessions.create(
            SessionRecord(
                id="session-cp",
                parent_id=None,
                status=SessionStatus.ACTIVE,
                version=1,
                created_at=now,
                updated_at=now,
            )
        )
        pause_events = await _collect(
            runtime.run_stream(
                spec,
                "call risky",
                session_id="session-cp",
                run_id="run-cp",
            )
        )
        assert any(e["type"] == "paused" for e in pause_events), pause_events
        paused = next(e for e in pause_events if e["type"] == "paused")
        await storage.approvals.approve(
            paused["approval_id"], expected_version=1, resolved_by="test"
        )
        resume_events = await _collect(runtime.resume("run-cp"))
        final = await storage.runs.get("run-cp")
        return resume_events, final

    resume_events, final_record = asyncio.run(_drive())
    assert resume_events[0]["type"] == "resumed"
    assert final_record.status is RunStatus.SUCCEEDED
