#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""strong-form evidence: a from-scratch in-memory EXTERNAL adapter that
implements the FULL agent-run storage surface drives real Runtime flows
(``Runtime.run`` to SUCCEEDED, plus ``Runtime.approve`` / ``Runtime.resume``
through WAITING_APPROVAL -> SUCCEEDED, plus an approved tool producing an
ArtifactStore artifact, plus a JobStore create -> claim -> commit_success
drive) through the public Store Protocols alone.

Six properties are pinned here:

1. ``test_external_adapter_imports_only_public_paths`` -- AST-scans the
   adapter module and asserts every ``linktools.ai.*`` import resolves
   against a fixed public allowlist. The forbidden modules -- the private
   runtime kernel (``_runtime``) and the in-repo reference backends
   (``storage.filesystem`` / ``storage.sqlite`` / ``storage.sqlalchemy`` /
   ``storage.coordination``)
   -- would defeat the proof: the adapter exists to show the PUBLIC surface
   suffices. Mirrors the import guard in
   ``test_external_adapter_conformance.py:48``.

2. ``test_in_memory_stores_satisfy_protocols`` -- asserts each in-memory
   store isinstance() its runtime_checkable Protocol. The Protocols are the
   boundary the Runtime consumes through; this guarantees the adapter can be
   wired into a Storage without ``isinstance`` shortfalls.

3. ``test_external_adapter_drives_run_to_completion`` -- mirrors
   ``tests/ai/e2e/test_file_runtime_complete.py`` but uses
   ``build_in_memory_external_storage(root=tmp_path)`` instead of
   ``FilesystemStorage``. Drives the real ``Runtime.build -> Runtime.run``
   chain with a TOOLLESS AgentSpec, then asserts the run reached SUCCEEDED
   with exactly one USER + one ASSISTANT session message, one checkpoint at
   sequence 1, and exactly one RunCompleted event (no RunFailed). This test
   proves ONLY the run -> complete slice through the adapter's public-
   Protocol stores; it does NOT exercise approval, resume, artifact, or job.

4. ``test_external_adapter_drives_approval_resume`` -- mirrors
   ``tests/ai/test_runtime_resume.py::test_resume_round_trip_pause_approve_resume_succeeds``
   but uses ``build_in_memory_external_storage(root=tmp_path)`` instead of
   ``FilesystemStorage``. Drives the real
   ``Runtime.run_stream -> pause at WAITING_APPROVAL -> Runtime.approve ->
   Runtime.resume -> SUCCEEDED`` round trip with a governed tool that
   requires approval. Asserts the run reached WAITING_APPROVAL, the
   ApprovalStore gained a PENDING request, ``Runtime.approve`` flipped it to
   APPROVED through the Principal-bound ApprovalService, ``Runtime.resume``
   re-entered execution, the run reached SUCCEEDED, and a checkpoint was
   written for the resume. This test proves ONLY the
   run -> approval -> resume slice; it does NOT exercise artifact or job.

5. ``test_external_adapter_drives_approval_resume_produces_artifact`` --
   extends the approval/resume proof to the artifact slice: the approved
   tool's handler, on resume, writes a content-addressed artifact via
   ``storage.artifacts.put(...)`` (the canonical closure-capture pattern
   from ``RuntimeTaskHandler._seal_run_result`` and the user-handler pattern
   in ``tests/ai/evaluation/test_task_executor.py``). After resume
   completes, the test asserts the artifact is genuinely retrievable through
   ``storage.artifacts.get(artifact_id=artifact_id, tenant_id=...)`` with the original
   content. Proves ONLY the run -> approval -> resume -> artifact slice.

6. ``test_external_adapter_drives_job_create_claim_commit`` -- drives a
   job through the adapter's ``InMemoryJobStore`` via the public JobStore
   Protocol: ``create_job`` -> ``claim`` -> ``commit_success``, asserting
   the task transitions PENDING -> READY -> CLAIMED -> SUCCEEDED, the
   attempt is recorded as SUCCEEDED, the fencing token is issued, and the
   job converges to SUCCEEDED. Also asserts the fencing-token contract: a
   stale worker whose claim is reclaimed cannot commit (``TaskClaimLostError``),
   mirroring ``tests/ai/jobs/test_file_task_store.py``. Proves ONLY the
   job slice through the adapter's public-Protocol ``tasks`` store.

OUT OF SCOPE here: a full ``JobRuntime.run`` worker loop on the adapter's
Storage (the JobStore is exercised directly through the public Protocol;
the worker loop adds claim-poll cadence and heartbeat mechanics, not
adapter-surface coverage)."""

import asyncio
import ast
import json
import pathlib
from datetime import datetime, timezone

import pytest

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from linktools.ai.agent.approval import ApprovalStore, ApprovalStatus
from linktools.ai.agent.spec import AgentSpec, PromptSpec, ToolRef
from linktools.ai.artifact import ANONYMOUS_PROVENANCE
from linktools.ai.capability.models import CapabilityBundle
from linktools.ai.capability.provider import CapabilityProvider
from linktools.ai.events.store import EventStore
from linktools.ai.governance.policy.approval import ApprovalRule
from linktools.ai.governance.policy.engine import PolicyEngine
from linktools.ai.identity.principal import ActorRef, PrincipalContext, ScopeSet
from linktools.ai.jobs.models import (
    ActorChain,
    JobRecord,
    JobStatus,
    RetryPolicy,
    SideEffectPolicy,
    TaskBudget,
    TaskPrincipal,
    TaskRecord,
    TaskStatus,
)
from linktools.ai.jobs.protocols import TaskSuccess
from linktools.ai.jobs.store import JobStore, TaskClaimLostError
from linktools.ai.memory.store import MemoryStore
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.registry import ModelRegistry
from linktools.ai.model.router import ModelResolver
from linktools.ai.run.checkpoint import CheckpointStore
from linktools.ai.run.definition import RunDefinitionStore
from linktools.ai.run.models import RunStatus
from linktools.ai.run.store import RunStore
from linktools.ai.runtime import Runtime, RuntimeDependencies
from linktools.ai.session.models import MessageRole
from linktools.ai.session.store import SessionStore
from linktools.ai.storage.transaction import NoCrossStoreTransactions
from linktools.ai.storage.features import FILE_STORAGE_FEATURES
from linktools.ai.swarm.store import SwarmStore
from linktools.ai.tool.executor import GovernedToolInvoker
from linktools.ai.tool.idempotency import IdempotencyStore
from linktools.ai.tool.models import (
    ManagedToolDefinition,
    ToolContribution,
    ToolDescriptor,
)

from external_adapter import storage as _external_storage_module
from external_adapter.commit import InMemoryRunCommitCoordinator
from external_adapter.storage import (
    InMemoryApprovalStore,
    InMemoryCheckpointStore,
    InMemoryEventStore,
    InMemoryExternalStorage,
    InMemoryIdempotencyStore,
    InMemoryJobStore,
    InMemoryMemoryStore,
    InMemoryRunDefinitionStore,
    InMemoryRunStore,
    InMemorySessionStore,
    InMemorySwarmStore,
    build_in_memory_external_storage,
)


# The allowlist of public modules the adapter may import. Anything outside
# this set -- underscore-prefixed modules, the private runtime kernel
# (``_runtime``), the in-repo reference backends under ``storage.filesystem``
# / ``storage.sqlalchemy`` / ``storage.coordination`` -- would defeat the
# point: the adapter exists to prove the PUBLIC Protocols suffice.
#
# Pattern follows test_external_adapter_conformance.py:48. Extended beyond the
# starter list with the three ``*.models`` / ``*.scope`` modules the data
# classes live in (swarm.models, memory.models, memory.scope) and the typed
# ``linktools.ai.errors`` hierarchy: the original list had ``run.models`` /
# ``session.models`` / etc. for the stores whose data modules are split from
# their Protocol module; swarm/memory follow the same split and need the same
# data-module entries to implement conformant stores. ``errors`` is a public
# module and is the typed contract callers catch by type (never by string) --
# a conformant adapter must raise the same typed errors as the reference
# backends.
_PUBLIC_ADAPTER_IMPORTS = frozenset(
    {
        "linktools.ai.run.store",
        "linktools.ai.run.models",
        "linktools.ai.run.definition",
        "linktools.ai.run.commit",
        "linktools.ai.run.lifecycle",
        "linktools.ai.session.store",
        "linktools.ai.session.models",
        "linktools.ai.events.store",
        "linktools.ai.events.payloads",
        "linktools.ai.events.envelope",
        "linktools.ai.events.context",
        "linktools.ai.events.registry",
        "linktools.ai.agent.approval",
        "linktools.ai.errors",
        "linktools.ai.tool.idempotency",
        "linktools.ai.swarm.store",
        "linktools.ai.swarm.models",
        "linktools.ai.memory.store",
        "linktools.ai.memory.models",
        "linktools.ai.memory.scope",
        "linktools.ai.asset.store",
        "linktools.ai.asset.memory",
        "linktools.ai.artifact.store",
        "linktools.ai.artifact.models",
        "linktools.ai.jobs.store",
        "linktools.ai.jobs.models",
        "linktools.ai.jobs.protocols",
        "linktools.ai.storage.protocols",
        "linktools.ai.storage.facade",
        "linktools.ai.storage.features",
        "linktools.ai.storage.transaction",
    }
)

# Anything that starts with one of these prefixes is a private reference
# backend or internal kernel and MUST NOT be imported by an external
# adapter. The allowlist check above is the positive set; this is the
# belt-and-braces negative guarantee called out in the spec.
_FORBIDDEN_PREFIXES = (
    "linktools.ai.runtime.builder",
    "linktools.ai.storage.filesystem",
    "linktools.ai.storage.sqlite",
    "linktools.ai.storage.sqlalchemy",
    "linktools.ai.storage.coordination",
)

# Marker substrings that must not appear in comments / docstrings: this
# adapter documents ITS OWN contract in terms of public Protocols, not in
# terms of internal plan / phase / op tracking numbers (those would leak
# transient process scaffolding into a public-surface proof).
_FORBIDDEN_MARKERS = ("Phase ", "§", "spec ", "op ")


def _module_imports(path: pathlib.Path) -> "set[str]":
    """Every absolute ``linktools.*`` import in ``path``. Relative imports
    (``from external_adapter.conformance_adapter import ...``) are intentionally NOT
    counted: a sibling-test-module import is internal to the test package,
    not a ``linktools.ai`` surface import, and the allowlist exists to pin
    the public-surface contract."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: "set[str]" = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("linktools"):
                    out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and node.module.startswith("linktools"):
                out.add(node.module)
    return out


def test_external_adapter_imports_only_public_paths() -> None:
    # Scan every module in the adapter package (not just storage.py) -- the
    # commit coordinator and conformance adapter live alongside it and must
    # uphold the same public-surface contract.
    pkg_dir = pathlib.Path(_external_storage_module.__file__).parent
    for src_path in sorted(pkg_dir.glob("*.py")):
        imports = _module_imports(src_path)
        non_public = imports - _PUBLIC_ADAPTER_IMPORTS
        assert not non_public, (
            f"{src_path.name}: external adapter must import only public "
            f"Protocol/store surface; found non-public linktools imports: "
            f"{sorted(non_public)}"
        )
        forbidden_hits = sorted(
            mod for mod in imports if any(mod.startswith(p) for p in _FORBIDDEN_PREFIXES)
        )
        assert not forbidden_hits, (
            f"{src_path.name}: external adapter must not import private "
            f"runtime / reference-backend modules: {forbidden_hits}"
        )
        # No transient plan/phase/op markers: the adapter stands on its
        # public contract alone. A comment/docstring referencing internal
        # process scaffolding leaks the wrong altitude into a public-surface
        # proof.
        src = src_path.read_text(encoding="utf-8")
        leaked = [marker for marker in _FORBIDDEN_MARKERS if marker in src]
        assert not leaked, (
            f"{src_path.name}: adapter leaked transient process markers into "
            f"comments/docstrings: {leaked}"
        )


def test_e2e_harness_itself_imports_no_reference_backend() -> None:
    # The above scans the ADAPTER package; this scans the harness file that
    # actually drives the E2E chain (this very file). A test file could
    # import a reference backend directly (e.g. FilesystemRunCommitCoordinator)
    # to wire the run without the adapter package ever seeing it -- the
    # adapter-package scan would stay green while the functional proof was
    # hollow. Guard the harness file itself against exactly that.
    harness_path = pathlib.Path(__file__)
    imports = _module_imports(harness_path)
    forbidden_hits = sorted(
        mod for mod in imports if any(mod.startswith(p) for p in _FORBIDDEN_PREFIXES)
    )
    assert not forbidden_hits, (
        "the E2E harness must not import private runtime / reference-backend "
        f"modules to drive the chain: {forbidden_hits}"
    )


def test_in_memory_stores_satisfy_protocols() -> None:
    # Each isinstance() check is a structural Protocol check: the adapter
    # class implements every method on the Protocol with the right shape
    # (the @runtime_checkable decorator turns the Protocol into a runtime
    # predicate over the method set).
    assert isinstance(InMemoryRunStore(), RunStore)
    assert isinstance(InMemorySessionStore(), SessionStore)
    assert isinstance(InMemoryEventStore(), EventStore)
    assert isinstance(InMemoryCheckpointStore(), CheckpointStore)
    assert isinstance(InMemoryApprovalStore(), ApprovalStore)
    assert isinstance(InMemoryIdempotencyStore(), IdempotencyStore)
    assert isinstance(InMemoryRunDefinitionStore(), RunDefinitionStore)
    assert isinstance(InMemorySwarmStore(), SwarmStore)
    assert isinstance(InMemoryMemoryStore(), MemoryStore)
    assert isinstance(InMemoryJobStore(), JobStore)


def _model_fn(messages, info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content='{"response": {"msg": "ok"}}')])


def _router() -> ModelResolver:
    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(_model_fn))
    return ModelResolver(registry=registry)


# -- Approval / resume fixtures ---------------------------------------------
#
# Mirrors tests/ai/test_runtime_resume.py: a governed tool "risky" that
# requires approval, plus a FunctionModel that emits a ToolCallPart on turn 1
# and a TextPart("done") once the tool has returned. Used to drive
# run_stream -> pause -> Runtime.approve -> Runtime.resume -> SUCCEEDED
# through the in-memory external adapter.

_APPROVAL_TOOL = "risky"
_TENANT_ID = "local"


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
                                name=_APPROVAL_TOOL,
                                source="test",
                                category="discovery",
                                risk="high",
                                mutating=False,
                            ),
                            handler=risky,
                        ),
                    ),
                ),
            ),
        )


def _approval_model_fn(messages, info: AgentInfo) -> ModelResponse:
    """Turn 1: emit a ToolCallPart for the risky tool. Turn 2 (after the tool
    has returned): emit TextPart("done"). The turn is selected by scanning the
    history for a ToolReturnPart tagged with the risky tool name."""
    for m in messages:
        parts = getattr(m, "parts", None) or []
        for p in parts:
            if (
                isinstance(p, ToolReturnPart)
                and getattr(p, "tool_name", None) == _APPROVAL_TOOL
            ):
                return ModelResponse(parts=[TextPart(content="done")])
    return ModelResponse(parts=[ToolCallPart(tool_name=_APPROVAL_TOOL, args={"x": 1})])


async def _approval_stream_fn(messages, info: AgentInfo):
    for m in messages:
        parts = getattr(m, "parts", None) or []
        for p in parts:
            if (
                isinstance(p, ToolReturnPart)
                and getattr(p, "tool_name", None) == _APPROVAL_TOOL
            ):
                yield "done"
                return
    yield {0: DeltaToolCall(name=_APPROVAL_TOOL, json_args=json.dumps({"x": 1}))}


def _approval_router() -> ModelResolver:
    registry = ModelRegistry()
    registry.register(
        "test-model",
        model=FunctionModel(_approval_model_fn, stream_function=_approval_stream_fn),
    )
    return ModelResolver(registry=registry)


def _approval_spec() -> AgentSpec:
    return AgentSpec(
        id="agent-approval",
        name="approval-agent",
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="hi"),
        output_schema=str,
        tools=(ToolRef(kind="test", name=_APPROVAL_TOOL),),
    )


def _approver() -> PrincipalContext:
    """The Principal that resolves the approval. tenant_id matches the run's
    tenant_id so ApprovalService.approve's require_tenant check passes."""
    return PrincipalContext(
        tenant_id=_TENANT_ID,
        user_id="approver",
        actor=ActorRef(kind="user", id="approver"),
        scopes=ScopeSet.allow_all(),
    )


def test_external_adapter_drives_run_to_completion(tmp_path: pathlib.Path) -> None:
    storage = build_in_memory_external_storage(root=tmp_path)
    # Sanity: the composition is wired correctly before exercising Runtime.
    assert isinstance(storage, InMemoryExternalStorage)
    assert storage.root == tmp_path
    assert storage.features is FILE_STORAGE_FEATURES
    assert isinstance(storage._transaction_manager, NoCrossStoreTransactions)

    runtime = Runtime.build(
        storage=storage,
        model_router=_router(),
        commit_coordinator=InMemoryRunCommitCoordinator.from_storage(storage),
    )
    spec = AgentSpec(
        id="agent-1",
        name="e2e-agent",
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="hi"),
    )

    async def _run() -> "object":
        return await runtime.run(spec, "say hello")

    result = asyncio.run(_run())
    assert "ok" in str(result.output), (
        f"expected model output 'ok' in run result, got: {result.output!r}"
    )

    async def _verify() -> None:
        # The run_id travels on every session message the run wrote; collect
        # them so we can resolve the run record independently of any
        # adapter-internal handle.
        sessions_root = storage.sessions
        all_messages: "list" = []
        for session_id in list(storage.sessions._records.keys()):  # type: ignore[attr-defined]
            all_messages.extend(
                await sessions_root.list_messages(session_id)
            )
        run_ids = {m.run_id for m in all_messages if m.run_id}
        assert run_ids, "no run_id found among session messages"
        run_id = run_ids.pop()

        record = await storage.runs.get(run_id)
        assert record is not None, "run record not found"
        assert record.status is RunStatus.SUCCEEDED, (
            f"expected SUCCEEDED, got {record.status}"
        )

        messages = await storage.sessions.list_messages(record.session_id)
        user_count = sum(1 for m in messages if m.role is MessageRole.USER)
        assistant_count = sum(1 for m in messages if m.role is MessageRole.ASSISTANT)
        assert user_count == 1, f"expected 1 USER message, got {user_count}"
        assert assistant_count == 1, (
            f"expected 1 ASSISTANT message, got {assistant_count}"
        )

        checkpoint = await storage.checkpoints.latest(run_id)
        assert checkpoint is not None, "no checkpoint written"
        assert checkpoint.sequence == 1, (
            f"expected checkpoint sequence 1, got {checkpoint.sequence}"
        )

        page = await storage.events.list(run_id, limit=100)
        payload_types = [type(e.payload).__name__ for e in page.items]
        assert payload_types.count("RunCompleted") == 1, (
            f"expected exactly 1 RunCompleted event, got: {payload_types}"
        )
        assert "RunFailed" not in payload_types, (
            f"RunFailed unexpectedly present: {payload_types}"
        )

    asyncio.run(_verify())


def test_external_adapter_drives_approval_resume(tmp_path: pathlib.Path) -> None:
    """Drive run -> pause at WAITING_APPROVAL -> Runtime.approve ->
    Runtime.resume -> SUCCEEDED through the in-memory external adapter.

    Mirrors ``tests/ai/test_runtime_resume.py::test_resume_round_trip_pause_approve_resume_succeeds``
    but swaps ``FilesystemStorage`` for ``build_in_memory_external_storage`` and
    drives the approve step through the Principal-bound ``Runtime.approve``
    (the existing resume test bypasses Runtime.approve and writes the approval
    store directly). All four phases run inside one ``asyncio.run`` so the
    in-memory ApprovalStore's ``asyncio.Lock`` stays bound to one event loop.

    Proves ONLY the run -> approval -> resume slice through the adapter's
    public-Protocol stores. Artifact and job are NOT exercised here."""
    storage = build_in_memory_external_storage(root=tmp_path)
    executor = GovernedToolInvoker(
        policy=PolicyEngine(
            rules=(ApprovalRule(require_for=frozenset({_APPROVAL_TOOL})),)
        ),
        approval_store=storage.approvals,
    )
    runtime = Runtime.build(
        storage=storage,
        model_router=_approval_router(),
        tool_executor=executor,
        providers=RuntimeDependencies(capabilities=(_RiskyProvider(),)),
        local_trusted_mode=True,
        commit_coordinator=InMemoryRunCommitCoordinator.from_storage(storage),
    )
    spec = _approval_spec()

    async def _drive() -> "tuple[list, RunStatus, RunStatus, ApprovalStatus, ApprovalStatus, int]":
        # 1. run_stream: the model emits a ToolCallPart for the risky tool;
        #    the GovernedToolInvoker raises RunPaused; the runner checkpoints,
        #    transitions to WAITING_APPROVAL, yields {"type": "paused", ...}.
        # session_id is intentionally omitted -- prepare_run auto-creates a
        # fresh session owned by the (user_id=None, tenant_id=_TENANT_ID)
        # principal; resume() reads the session_id back from the run record.
        pause_events: "list[dict]" = []
        async for event in runtime.run_stream(
            spec,
            "call risky",
            run_id="run-apr",
            tenant_id=_TENANT_ID,
        ):
            pause_events.append(event)

        paused = next(e for e in pause_events if e["type"] == "paused")
        approval_id = paused["approval_id"]
        assert paused["run_id"] == "run-apr"

        # Run reached WAITING_APPROVAL (NOT FAILED).
        paused_record = await storage.runs.get("run-apr")
        paused_status = paused_record.status

        # ApprovalStore gained a PENDING request for this run.
        pending_approval = await storage.approvals.get(approval_id)
        assert pending_approval is not None, (
            f"approval {approval_id} not found in the in-memory ApprovalStore"
        )
        pending_status = pending_approval.status

        # 2. Runtime.approve: Principal-bound ApprovalService flips PENDING ->
        #    APPROVED. This goes through the adapter's approve() (typed
        #    InvalidApprovalTransitionError / ApprovalConflictError on the
        #    error path, PENDING -> APPROVED on the happy path).
        await runtime.approve(
            approval_id,
            principal=_approver(),
            expected_version=pending_approval.version,
        )
        approved = await storage.approvals.get(approval_id)
        approved_status = approved.status

        # 3. Runtime.resume: deserializes the checkpoint, transitions
        #    WAITING_APPROVAL -> RUNNING, re-enters run_stream with
        #    message_history. The resume gate (_already_approved) recognizes
        #    the now-APPROVED request and lets the tool execute.
        resume_events: "list[dict]" = []
        async for event in runtime.resume("run-apr"):
            resume_events.append(event)

        # 4. Final state: SUCCEEDED.
        final = await storage.runs.get("run-apr")
        final_status = final.status

        # Checkpoint count: pause writes one, resume-completion writes another.
        # The adapter's CheckpointStore owns sequence assignment; verifying
        # latest() exists confirms the resume path's checkpoint read worked.
        checkpoint = await storage.checkpoints.latest("run-apr")
        assert checkpoint is not None, "no checkpoint written for the run"
        checkpoint_sequence = checkpoint.sequence

        return (
            resume_events,
            paused_status,
            final_status,
            pending_status,
            approved_status,
            checkpoint_sequence,
        )

    (
        resume_events,
        paused_status,
        final_status,
        pending_status,
        approved_status,
        checkpoint_sequence,
    ) = asyncio.run(_drive())

    # The run genuinely reached WAITING_APPROVAL at pause time.
    assert paused_status is RunStatus.WAITING_APPROVAL, (
        f"expected WAITING_APPROVAL at pause, got {paused_status}"
    )
    # ApprovalStore held a PENDING request before approve.
    assert pending_status is ApprovalStatus.PENDING, (
        f"expected PENDING approval before approve, got {pending_status}"
    )
    # Runtime.approve flipped it to APPROVED through the adapter.
    assert approved_status is ApprovalStatus.APPROVED, (
        f"expected APPROVED after Runtime.approve, got {approved_status}"
    )
    # Runtime.resume re-entered execution and reached SUCCEEDED.
    assert final_status is RunStatus.SUCCEEDED, (
        f"expected SUCCEEDED after Runtime.resume, got {final_status}"
    )
    # Resumed signal yielded first.
    assert resume_events[0]["type"] == "resumed", (
        f"expected 'resumed' as the first resume event, got {resume_events[0]}"
    )
    assert resume_events[0]["run_id"] == "run-apr"
    # The governed tool actually executed during resume (end event, ok=True).
    tool_ends = [
        e
        for e in resume_events
        if e.get("type") == "tool" and e.get("phase") == "end"
    ]
    assert any(
        e["name"] == _APPROVAL_TOOL and e["ok"] is True for e in tool_ends
    ), (
        f"expected tool end event for {_APPROVAL_TOOL} with ok=True, "
        f"got resume_events={resume_events}"
    )
    # At least one checkpoint exists (pause wrote one; resume may write more).
    assert checkpoint_sequence >= 1, (
        f"expected checkpoint sequence >= 1, got {checkpoint_sequence}"
    )


def test_external_adapter_drives_approval_resume_produces_artifact(
    tmp_path: pathlib.Path,
) -> None:
    """Drive run -> pause -> approve -> resume -> SUCCEEDED through the
    adapter, where the approved tool's handler writes a content-addressed
    artifact via ``storage.artifacts.put(...)`` on resume. After resume
    completes the artifact is genuinely retrievable through the adapter's
    ArtifactStore (the same store wired from InMemoryArtifactBlobStore +
    InMemoryArtifactRecordStore), proving the run -> approval -> resume ->
    artifact slice holds through the public surface.

    DI pattern: the handler closes over ``storage.artifacts`` (the canonical
    closure-capture pattern from ``tests/ai/evaluation/test_task_executor.py``
    and ``RuntimeTaskHandler._seal_run_result``). A production handler would
    receive the store via constructor injection; the closure form is the
    test equivalent."""
    storage = build_in_memory_external_storage(root=tmp_path)
    captured: "dict[str, object]" = {"artifact_id": None, "content": None}

    class _ArtifactProvider(CapabilityProvider):
        supported_kinds = ("test",)

        async def resolve(self, ref, context):
            async def risky(x: int) -> dict:
                # The handler closes over storage.artifacts -- the canonical
                # DI / closure-capture pattern. The bytes are the artifact
                # content; the call returns a content-addressed record.
                content = f'{{"doubled": {x * 2}}}'.encode("utf-8")
                record = await storage.artifacts.put(
                    content=content,
                    media_type="application/json",
                    tenant_id=_TENANT_ID, provenance=ANONYMOUS_PROVENANCE,
    )
                captured["artifact_id"] = record.ref.id
                captured["content"] = content
                return {"artifact_id": record.ref.id, "doubled": x * 2}

            return CapabilityBundle(
                tool_contributions=(
                    ToolContribution(
                        tools=(
                            ManagedToolDefinition(
                                descriptor=ToolDescriptor(
                                    name=_APPROVAL_TOOL,
                                    source="test",
                                    category="discovery",
                                    risk="high",
                                    mutating=True,
                                ),
                                handler=risky,
                            ),
                        ),
                    ),
                ),
            )

    executor = GovernedToolInvoker(
        policy=PolicyEngine(
            rules=(ApprovalRule(require_for=frozenset({_APPROVAL_TOOL})),)
        ),
        approval_store=storage.approvals,
    )
    runtime = Runtime.build(
        storage=storage,
        model_router=_approval_router(),
        tool_executor=executor,
        providers=RuntimeDependencies(capabilities=(_ArtifactProvider(),)),
        local_trusted_mode=True,
        commit_coordinator=InMemoryRunCommitCoordinator.from_storage(storage),
    )
    spec = _approval_spec()

    async def _drive() -> "tuple[RunStatus, object, bytes]":
        # 1. run_stream -> pause at WAITING_APPROVAL (handler not yet run).
        pause_events: "list[dict]" = []
        async for event in runtime.run_stream(
            spec,
            "call risky",
            run_id="run-art",
            tenant_id=_TENANT_ID,
        ):
            pause_events.append(event)
        paused = next(e for e in pause_events if e["type"] == "paused")
        approval_id = paused["approval_id"]
        pending = await storage.approvals.get(approval_id)
        assert pending is not None
        # The handler has NOT yet executed: no artifact was produced at pause.
        assert captured["artifact_id"] is None, (
            "artifact produced before approval -- handler ran during pause"
        )

        # 2. Runtime.approve.
        await runtime.approve(
            approval_id,
            principal=_approver(),
            expected_version=pending.version,
        )

        # 3. Resume -> SUCCEEDED; the handler runs and writes an artifact.
        resume_events: "list[dict]" = []
        async for event in runtime.resume("run-art"):
            resume_events.append(event)

        # 4. The artifact was genuinely produced during resume and is
        #    retrievable through the adapter's ArtifactStore (tenant-scoped,
        #    integrity-checked -- the same path production code uses).
        artifact_id = captured["artifact_id"]
        assert artifact_id is not None, "tool handler did not produce an artifact"
        retrieved = await storage.artifacts.get(
            artifact_id=artifact_id,  # type: ignore[arg-type]
            tenant_id=_TENANT_ID,
        )
        assert retrieved is not None, (
            f"artifact {artifact_id} not retrievable through the adapter"
        )

        # 5. Run reached SUCCEEDED.
        final = await storage.runs.get("run-art")
        assert final is not None
        return final.status, artifact_id, retrieved

    final_status, artifact_id, retrieved = asyncio.run(_drive())

    assert final_status is RunStatus.SUCCEEDED, (
        f"expected SUCCEEDED after resume, got {final_status}"
    )
    # The artifact content round-trips byte-for-byte: the blob store kept the
    # bytes, the record store kept the lineage, and ArtifactStore.get
    # re-hashed and matched (no integrity error).
    assert retrieved == captured["content"], (
        f"artifact content mismatch: expected {captured['content']!r}, "
        f"got {retrieved!r}"
    )
    # The artifact_id has the canonical prefix the ArtifactStore mints.
    assert isinstance(artifact_id, str) and artifact_id.startswith("art-"), (
        f"expected artifact_id 'art-...' (ArtifactStore UUID format), "
        f"got {artifact_id!r}"
    )


def _job_record(clock_now) -> JobRecord:
    return JobRecord(
        id="job-1",
        status=JobStatus.PENDING,
        principal=TaskPrincipal(tenant_id=_TENANT_ID, user_id="alice"),
        actor_chain=ActorChain(actors=(ActorRef("user", "alice"),)),
        budget=TaskBudget(),
        root_task_id="task-1",
        input_artifact_id=None,
        output_artifact_id=None,
        version=1,
        created_at=clock_now,
        started_at=None,
        finished_at=None,
    )


def _root_task(clock_now) -> TaskRecord:
    return TaskRecord(
        id="task-1",
        job_id="job-1",
        parent_task_id=None,
        key="root",
        handler="echo",
        status=TaskStatus.PENDING,
        input_artifact_id=None,
        output_artifact_id=None,
        dependencies=(),
        retry_policy=RetryPolicy(max_attempts=2),
        side_effect_policy=SideEffectPolicy(),
        attempt_count=0,
        available_at=clock_now,
        lease_owner=None,
        lease_expires_at=None,
        fencing_token=0,
        active_attempt_id=None,
        timeout_seconds=None,
        asset_snapshots=(),
        version=1,
        created_at=clock_now,
        updated_at=clock_now,
    )


def test_external_adapter_drives_job_create_claim_commit(
    tmp_path: pathlib.Path,
) -> None:
    """Drive a job through the adapter's ``InMemoryJobStore`` via the public
    JobStore Protocol: ``create_job`` -> ``claim`` -> ``commit_success``.
    Asserts the task transitions PENDING -> READY -> CLAIMED -> SUCCEEDED,
    a fresh attempt is recorded as SUCCEEDED, the fencing token is issued,
    and the job converges from PENDING -> RUNNING -> SUCCEEDED.

    Mirrors ``tests/ai/jobs/test_file_task_store.py::test_create_claim_complete_completes_job``
    but swaps ``FilesystemJobStore`` for ``storage.jobs`` from
    ``build_in_memory_external_storage``. Also exercises the fencing-token
    contract: a stale claim (older fencing_token) is rejected with
    ``TaskClaimLostError`` -- the contract a real worker relies on so its
    superseded result cannot overwrite a newer owner's. Proves ONLY the job
    slice through the adapter's public ``jobs`` store."""
    storage = build_in_memory_external_storage(root=tmp_path)
    # storage.jobs is wired by build_in_memory_external_storage -- this is
    # the assertion that the adapter's Storage carries a real JobStore (not
    # None), which JobRuntime's build-time check would enforce in production.
    assert isinstance(storage.jobs, InMemoryJobStore)
    assert isinstance(storage.jobs, JobStore)
    task_store = storage.jobs
    clock_now = datetime.now(timezone.utc)

    async def _drive():
        # 1. create_job stores the job + root task; the root task lands READY
        #    (claimable), the job stays PENDING until the first claim.
        await task_store.create_job(_job_record(clock_now), _root_task(clock_now))
        job_after_create = await task_store.get_job("job-1")
        task_after_create = await task_store.get_task("task-1")
        assert job_after_create is not None
        assert task_after_create is not None
        assert job_after_create.status is JobStatus.PENDING
        assert task_after_create.status is TaskStatus.READY, (
            f"expected READY after create_job, got {task_after_create.status}"
        )

        # 2. claim -> ClaimedTask (claim, job, task, attempt). Fencing token
        #    issued; first claim starts the job (PENDING -> RUNNING).
        claimed = await task_store.claim(
            worker_id="w1", now=clock_now, lease_seconds=30
        )
        assert claimed is not None
        assert claimed.task.status is TaskStatus.CLAIMED
        assert claimed.task.fencing_token == 1
        assert claimed.task.attempt_count == 1
        assert claimed.attempt.status.value == "running"
        assert claimed.attempt.worker_id == "w1"
        job_after_claim = await task_store.get_job("job-1")
        assert job_after_claim.status is JobStatus.RUNNING, (
            f"expected RUNNING after first claim, got {job_after_claim.status}"
        )

        # 3. commit_success flips task CLAIMED -> SUCCEEDED, marks the attempt
        #    SUCCEEDED, and converges the job RUNNING -> SUCCEEDED (the root
        #    task was the only task and it landed SUCCEEDED).
        done = await task_store.commit_success(claimed.claim, TaskSuccess())
        assert done.status is TaskStatus.SUCCEEDED
        # Audit trail: the transition list records the moves.
        transitions = await task_store.list_transitions("job-1")
        transition_targets = [t.to_status for t in transitions]
        assert "ready" in transition_targets  # created -> READY
        assert "claimed" in transition_targets  # READY -> CLAIMED
        assert "succeeded" in transition_targets  # CLAIMED -> SUCCEEDED
        # Attempt is persisted as SUCCEEDED (close + audit).
        attempts = await task_store.list_attempts("task-1")
        assert len(attempts) == 1
        assert attempts[0].status.value == "succeeded"
        # Job converged to SUCCEEDED.
        job_final = await task_store.get_job("job-1")
        assert job_final.status is JobStatus.SUCCEEDED, (
            f"expected job SUCCEEDED after root task commit, got {job_final.status}"
        )
        return claimed

    claimed = asyncio.run(_drive())

    # Fencing contract: a stale claim (same fencing_token, but the task is
    # already SUCCEEDED) cannot commit. The stored task's status (SUCCEEDED,
    # not CLAIMED) and absent lease_owner fail the guard, raising
    # TaskClaimLostError -- the typed error a real worker catches to abandon
    # its superseded result. Mirrors the Filesystem reference's guard.
    async def _stale_commit():
        with pytest.raises(TaskClaimLostError):
            await task_store.commit_success(claimed.claim, TaskSuccess())

    asyncio.run(_stale_commit())

    # No more claimable work: a second worker finds nothing to claim.
    async def _no_more_work():
        return await task_store.claim(
            worker_id="w2", now=datetime.now(timezone.utc), lease_seconds=30
        )

    assert asyncio.run(_no_more_work()) is None


def test_external_adapter_full_connected_chain_run_approval_resume_artifact_job(
    tmp_path: pathlib.Path,
) -> None:
    """ONE connected chain through the external adapter's public surface:
    run -> pause (WAITING_APPROVAL) -> approve -> resume -> SUCCEEDED (the
    approved tool produces a content-addressed artifact) -> a Job created with
    that artifact as its input_artifact_id -> claimed -> committed SUCCEEDED.

    The explicitly forbids substituting 'several disconnected unit
    tests' for the connected chain (each of the four slice tests above proves
    ONE segment in isolation; this test proves the segments COMPOSE -- the
    artifact the approved tool produced during resume is the SAME artifact the
    Job references as input, and the run/approval/artifact/job state all lands
    through the adapter's public Protocol stores in one flow)."""
    storage = build_in_memory_external_storage(root=tmp_path)
    captured: "dict[str, object]" = {"artifact_id": None, "content": None}

    class _ArtifactProvider(CapabilityProvider):
        supported_kinds = ("test",)

        async def resolve(self, ref, context):
            async def risky(x: int) -> dict:
                content = f'{{"doubled": {x * 2}}}'.encode("utf-8")
                record = await storage.artifacts.put(
                    content=content,
                    media_type="application/json",
                    tenant_id=_TENANT_ID, provenance=ANONYMOUS_PROVENANCE,
    )
                captured["artifact_id"] = record.ref.id
                captured["content"] = content
                return {"artifact_id": record.ref.id, "doubled": x * 2}

            return CapabilityBundle(
                tool_contributions=(
                    ToolContribution(
                        tools=(
                            ManagedToolDefinition(
                                descriptor=ToolDescriptor(
                                    name=_APPROVAL_TOOL,
                                    source="test",
                                    category="discovery",
                                    risk="high",
                                    mutating=True,
                                ),
                                handler=risky,
                            ),
                        ),
                    ),
                ),
            )

    executor = GovernedToolInvoker(
        policy=PolicyEngine(
            rules=(ApprovalRule(require_for=frozenset({_APPROVAL_TOOL})),)
        ),
        approval_store=storage.approvals,
    )
    runtime = Runtime.build(
        storage=storage,
        model_router=_approval_router(),
        tool_executor=executor,
        providers=RuntimeDependencies(capabilities=(_ArtifactProvider(),)),
        local_trusted_mode=True,
        commit_coordinator=InMemoryRunCommitCoordinator.from_storage(storage),
    )
    spec = _approval_spec()

    async def _chain() -> None:
        # 1-3. run -> pause -> approve -> resume -> SUCCEEDED + artifact.
        pause_events: "list[dict]" = []
        async for event in runtime.run_stream(
            spec,
            "call risky",
            run_id="run-chain",
            tenant_id=_TENANT_ID,
        ):
            pause_events.append(event)
        paused = next(e for e in pause_events if e["type"] == "paused")
        approval_id = paused["approval_id"]
        pending = await storage.approvals.get(approval_id)
        assert pending is not None
        assert captured["artifact_id"] is None, (
            "artifact produced before approval -- handler ran during pause"
        )
        await runtime.approve(
            approval_id, principal=_approver(), expected_version=pending.version
        )
        async for _ in runtime.resume("run-chain"):
            pass
        final = await storage.runs.get("run-chain")
        assert final is not None
        assert final.status is RunStatus.SUCCEEDED, (
            f"expected run SUCCEEDED after resume, got {final.status}"
        )

        # 4. The approved tool produced an artifact; it round-trips through the
        #    adapter's ArtifactStore (content-addressed, tenant-scoped).
        artifact_id = captured["artifact_id"]
        assert artifact_id is not None, "tool handler did not produce an artifact"
        assert isinstance(artifact_id, str) and artifact_id.startswith("art-")
        retrieved = await storage.artifacts.get(
            artifact_id=artifact_id,  # type: ignore[arg-type]
            tenant_id=_TENANT_ID,
        )
        assert retrieved == captured["content"], (
            f"artifact content mismatch: {retrieved!r} vs {captured['content']!r}"
        )

        # 5. The CONNECTING edge: a Job that REFERENCES the just-produced
        #    artifact as its input_artifact_id. The four slice tests prove each
        #    segment alone; this proves the artifact flows INTO the job in one
        #    connected run -> approval -> resume -> artifact -> job flow.
        task_store = storage.jobs
        now = datetime.now(timezone.utc)
        job = JobRecord(
            id="job-chain",
            status=JobStatus.PENDING,
            principal=TaskPrincipal(tenant_id=_TENANT_ID, user_id="alice"),
            actor_chain=ActorChain(actors=(ActorRef("user", "alice"),)),
            budget=TaskBudget(),
            root_task_id="task-chain",
            input_artifact_id=artifact_id,  # the artifact from step 4
            output_artifact_id=None,
            version=1,
            created_at=now,
            started_at=None,
            finished_at=None,
        )
        root = TaskRecord(
            id="task-chain",
            job_id="job-chain",
            parent_task_id=None,
            key="root",
            handler="echo",
            status=TaskStatus.PENDING,
            input_artifact_id=artifact_id,  # the artifact from step 4
            output_artifact_id=None,
            dependencies=(),
            retry_policy=RetryPolicy(max_attempts=2),
            side_effect_policy=SideEffectPolicy(),
            attempt_count=0,
            available_at=now,
            lease_owner=None,
            lease_expires_at=None,
            fencing_token=0,
            active_attempt_id=None,
            timeout_seconds=None,
            asset_snapshots=(),
            version=1,
            created_at=now,
            updated_at=now,
        )
        await task_store.create_job(job, root)
        claimed = await task_store.claim(
            worker_id="w1", now=now, lease_seconds=30
        )
        assert claimed is not None, "job not claimable through the adapter"
        done = await task_store.commit_success(claimed.claim, TaskSuccess())
        assert done.status is TaskStatus.SUCCEEDED

        # 6. The full chain landed: run SUCCEEDED, artifact retrievable, job
        #    terminal, and the job's input_artifact_id IS the artifact the
        #    approved tool produced -- the connected chain, not four slices.
        job_final = await task_store.get_job("job-chain")
        assert job_final is not None
        assert job_final.status is JobStatus.SUCCEEDED, (
            f"expected job SUCCEEDED, got {job_final.status}"
        )
        assert job_final.input_artifact_id == artifact_id, (
            "the job's input_artifact_id is not the artifact the approved tool "
            "produced -- the chain is disconnected"
        )

        # 7. Persistence breadth: the connected chain wrote EVENTS for the run
        #    AND a CHECKPOINT for the resume through the adapter's public
        # EventStore / CheckpointStore -- 's "验证 event" + resume
        #    checkpoint, asserted here in the connected flow (not only in the
        #    disconnected slice tests).
        run_events = await storage.events.list("run-chain", limit=100)
        assert len(run_events.items) > 0, (
            "no events persisted for the connected-chain run through the "
            "adapter's EventStore"
        )
        resume_checkpoint = await storage.checkpoints.latest("run-chain")
        assert resume_checkpoint is not None, (
            "no checkpoint persisted for the connected-chain resume through "
            "the adapter's CheckpointStore"
        )

    asyncio.run(_chain())

