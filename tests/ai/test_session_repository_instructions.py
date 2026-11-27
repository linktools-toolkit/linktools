#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session scope memory refreshes content without weakening execution pins."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.asset import AssetKey, AssetStore, InMemoryAssetBackend
from linktools.ai.capability import ExecutionCapability, SessionCapability, StepPersistenceCapability
from linktools.ai.core import ApprovalDecision, ApprovalStatus, CommitPolicy, ExecutionStatus, Principal, SessionStatus, WorkspaceFileInput
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.persistence import Persistence, PersistencePlan, PersistenceStorage
from linktools.ai.persistence.codec import decode_domain, encode_domain
from linktools.ai.runtime import Runtime
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.execution import ApprovalDecisionRequest, ExecutionRecoveryDetail
from linktools.ai.spec import RepositoryInstructionDocument, RepositoryInstructions
from linktools.ai.storage import InMemoryCacheBackend, StorageOverlay, StorageRoute
from linktools.ai.session import SessionRecord
from linktools.ai.workspace import ToolPermissionPolicy, ToolPermissionRule, Workspace, WorkspacePolicy

from ._runtime_test_helpers import (
    _Models,
    _UsageFunctionModel,
    _wait_for_committed,
    is_tool_retry,
    model_features,
    native_checkpoint,
    workspace_features,
)
from .test_canonical_turn_subagents import _HostAuthorization


@asynccontextmanager
async def _storage(root: Path, backend: str) -> AsyncIterator[PersistenceStorage]:
    engine = None
    if backend == "memory":
        storage = PersistenceStorage.in_memory()
    elif backend == "sqlite":
        storage = PersistenceStorage.sqlite(root / "state.db")
    elif backend == "split":
        storage = PersistenceStorage.from_plan(PersistencePlan(
            execution=StorageRoute.sqlite(root / "execution.db"),
            conversation=StorageRoute.sqlite(root / "conversation.db"),
        ))
    else:
        engine = create_async_engine(f"sqlite+aiosqlite:///{root / 'state.db'}")
        await provision_database(engine)
        storage = PersistenceStorage.sql(engine)
    try:
        yield storage
    finally:
        if engine is not None:
            await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("overflow", (None, "model", "turn"))
async def test_native_session_reads_new_scope_on_first_call_and_loads_rules(tmp_path: Path, overflow) -> None:
    nested = tmp_path / "pkg" / "deep"
    nested.mkdir(parents=True)
    (nested / "AGENTS.md").write_text("NESTED-ONLY-GUIDANCE", encoding="utf-8")
    (nested / "README.md").write_text("first read body", encoding="utf-8")
    if overflow is not None:
        (nested.parent / "AGENTS.md").write_text("OVER-BUDGET-" + "x" * (70 * 1024), encoding="utf-8")
    requests = 0

    async def model(messages, info):
        nonlocal requests
        requests += 1
        instructions = info.instructions or ""
        if requests == 1:
            assert "NESTED-ONLY-GUIDANCE" not in instructions
            return ModelResponse(parts=[
                ToolCallPart("read_file", {"path": "pkg/deep/README.md"}, "read-once"),
                ToolCallPart("list_files", {"path": "pkg/deep"}, "list-once"),
            ])
        assert requests == 2
        assert "NESTED-ONLY-GUIDANCE" in instructions
        assert "OVER-BUDGET-" not in instructions
        returns = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
        assert {part.tool_call_id for part in returns} == {"read-once", "list-once"}
        assert all(part.outcome == "success" for part in returns)
        assert "first read body" in str(next(part for part in returns if part.tool_call_id == "read-once").content)
        return ModelResponse(parts=[TextPart("done")])

    policy = WorkspacePolicy(max_repository_instruction_bytes=4096) if overflow == "model" else WorkspacePolicy()
    features = workspace_features("native-no-instruction-retry", workspace=Workspace.load(tmp_path, policy=policy))
    features.capability(SessionCapability(checkpoints=True), scope="runtime")
    features.agent("reader", model="default", allow_tools=("read_file", "list_files"), tool_retries=0)
    model_features(_Models(_UsageFunctionModel(model)), features)
    async with Persistence.open(
        "native-no-instruction-retry", storage=PersistenceStorage.sqlite(tmp_path / "state.db"),
    ) as persistence, Runtime.open(
        persistence=persistence, features=features, authorization=_HostAuthorization(),
    ) as runtime:
        agent = runtime.agents.get("reader")
        principal = runtime.default_principal
        await runtime.sessions.create_turn_session("session", principal=principal, idempotency_key="create")
        turn = await runtime.sessions.start(
            "session", "Read the file", agent=agent, principal=principal, idempotency_key="once",
        )
        assert (await turn.wait_final(timeout_seconds=10)).status is ExecutionStatus.SUCCEEDED
        assert requests == 2
        replay = await runtime.sessions.create_turn_session("session", principal=principal, idempotency_key="create")
        assert replay["session_id"] == "session" and replay["active_turn_id"] is None
        with pytest.raises(AIError) as different:
            await runtime.sessions.create_turn_session(
                "session", principal=principal, idempotency_key="create", metadata={"other": True},
            )
        assert different.value.code is ErrorCode.SESSION_CONFLICT


@pytest.mark.asyncio
async def test_native_workspace_approval_resumes_after_sibling_discovers_rules(tmp_path: Path) -> None:
    nested = tmp_path / "pkg"
    nested.mkdir()
    (nested / "AGENTS.md").write_text("SIBLING-GUIDANCE", encoding="utf-8")
    (tmp_path / "README.md").write_text("approved original body", encoding="utf-8")
    requests = 0

    async def model(messages, info):
        nonlocal requests
        requests += 1
        if requests == 1:
            assert "SIBLING-GUIDANCE" not in (info.instructions or "")
            return ModelResponse(parts=[
                ToolCallPart("read_file", {"path": "README.md"}, "approved-original"),
                ToolCallPart("list_files", {"path": "pkg"}, "discover-rules"),
            ])
        assert requests == 2
        assert "SIBLING-GUIDANCE" in (info.instructions or "")
        [result] = [part for message in messages for part in message.parts
                    if isinstance(part, ToolReturnPart) and part.tool_call_id == "approved-original"]
        assert result.outcome == "success" and "approved original body" in str(result.content)
        return ModelResponse(parts=[TextPart("done")])

    workspace = Workspace.load(tmp_path, policy=WorkspacePolicy(tool_permissions=ToolPermissionPolicy(
        rules=(ToolPermissionRule(tool_name="read_file", decision="ask"),),
    )))
    features = workspace_features("approval-after-discovery", workspace=workspace)
    features.capability(SessionCapability(checkpoints=True), scope="runtime")
    features.agent("reader", model="default", allow_tools=("read_file", "list_files"), tool_retries=0)
    model_features(_Models(_UsageFunctionModel(model)), features)
    async with Persistence.open(
        "approval-after-discovery", storage=PersistenceStorage.sqlite(tmp_path / "state.db"),
    ) as persistence, Runtime.open(
        persistence=persistence, features=features, authorization=_HostAuthorization(),
    ) as runtime:
        principal = Principal("operator", runtime.tenant_id)
        agent = runtime.agents.get("reader")
        await agent.create_session("session", principal=principal)
        turn = await runtime.sessions.start(
            "session", "Read the file", agent=agent, principal=principal, idempotency_key="once",
        )
        assert (await asyncio.wait_for(turn.wait(), timeout=10)).status is ExecutionStatus.WAITING_DEFERRED
        [pending] = await runtime.approvals.list(turn.key, principal=principal)
        await runtime.approvals.decide(turn.key, ApprovalDecisionRequest(
            principal, pending.approval_id, "approve-exact", ApprovalDecision.APPROVE,
        ))
        assert (await turn.wait_final(timeout_seconds=10)).status is ExecutionStatus.SUCCEEDED
        assert requests == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite", "sql", "split"))
@pytest.mark.parametrize("capture_frontier", (False, True))
async def test_session_reloads_known_scopes_before_first_model_request(
    tmp_path: Path, backend: str, capture_frontier: bool,
) -> None:
    # Backend/checkpoint coverage exercises durable scope reload and union.
    # Filesystem mutations and new-session isolation share the same implementation.
    native = backend != "memory"
    split_selects_step_only = backend == "split" and capture_frontier
    full_mutation_path = backend == "sql" and capture_frontier
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    for scope in ("pkg", "other"):
        directory = workspace_root / scope
        directory.mkdir()
        (directory / "file.txt").write_text(f"{scope} contents", encoding="utf-8")
    rules = workspace_root / "pkg" / "AGENTS.md"
    rules.write_text("PACKAGE-V1", encoding="utf-8")
    (workspace_root / "other" / "AGENTS.md").write_text("OTHER-RULE", encoding="utf-8")
    turn = ""
    attempts = 0
    target = "pkg/file.txt"
    initial_rule: str | None = None
    discovery_rule: str | None = None
    forbidden_rules: tuple[str, ...] = ()

    async def model(messages, info):
        nonlocal attempts
        attempts += 1
        instructions = info.instructions or ""
        if attempts == 1:
            if initial_rule is not None:
                assert initial_rule in instructions
            for text in forbidden_rules:
                assert text not in instructions
            if discovery_rule is not None:
                assert discovery_rule not in instructions
        else:
            if discovery_rule is not None:
                assert discovery_rule in instructions
            assert not any(is_tool_retry(part) for part in messages[-1].parts)
            assert "contents" in str(messages[-1])
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart(
            "read_file", {"path": target}, f"{turn}-{attempts}",
        )])

    features = workspace_features("scope-memory", workspace=Workspace.load(workspace_root))
    if native:
        features.capability(SessionCapability(checkpoints=capture_frontier and not split_selects_step_only), scope="runtime")
    else:
        features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.agent("reader", model="default", allow_tools=("read_file",), allow_subagents=())
    features = model_features(_Models(_UsageFunctionModel(model)), features)

    async def run(runtime, session_id: str, name: str, *, initial: str | None = None,
                  discover: str | None = None, absent: tuple[str, ...] = ()) -> None:
        nonlocal turn, attempts, initial_rule, discovery_rule, forbidden_rules
        turn, attempts = name, 0
        initial_rule, discovery_rule, forbidden_rules = initial, discover, absent
        if native:
            try:
                handle = await runtime.sessions.start(
                    session_id, name, agent="reader", principal=runtime.default_principal, idempotency_key=name,
                )
            except AIError as error:
                pytest.fail(f"native admission failed: {error.code} {error.safe_details}")
            result = await handle.wait_final(timeout_seconds=10)
        else:
            session = runtime.agents.get("reader").session(session_id)
            result = (await session.run(name, timeout_seconds=10)).result
        assert result.status is ExecutionStatus.SUCCEEDED
        assert attempts == 2

    async with _storage(tmp_path, backend) as storage:
        async with Persistence.open("scope-memory", storage=storage) as persistence:
            if capture_frontier and not native:
                features.capability(native_checkpoint(database=tmp_path / "native.db"), scope="runtime")

            auth = {"authorization": _HostAuthorization()} if native else {}

            async def assert_scope(session_id, expected):
                if native:
                    stored = await persistence.session_turns.get_session(session_id)
                    assert tuple(stored["config_json"].get("instruction_scopes", ())) == expected
                    assert "PACKAGE-V1" not in str(stored["config_json"])
                else:
                    stored = await persistence.conversation.sessions.get(session_id, tenant_id=persistence.tenant_id)
                    assert stored is not None and stored.repository_instruction_scopes == expected
                    assert "PACKAGE-V1" not in str(encode_domain(stored))

            async with Runtime.open(persistence=persistence, features=features, **auth) as runtime:
                await runtime.agents.get("reader").create_session("session")
                await run(runtime, "session", "learn", discover="PACKAGE-V1")
                await assert_scope("session", ("pkg",))

            if full_mutation_path:
                async with Runtime.open(persistence=persistence, features=features, **auth) as runtime:
                    await run(runtime, "session", "same", initial="PACKAGE-V1")

            # Reload the scope, not the previous execution's content pin, before
            # the first model request in the new Runtime.
            rules.write_text("PACKAGE-V2", encoding="utf-8")
            async with Runtime.open(persistence=persistence, features=features, **auth) as runtime:
                await run(runtime, "session", "changed", initial="PACKAGE-V2", absent=("PACKAGE-V1",))
                current_rule = "PACKAGE-V2"
                if full_mutation_path:
                    rules.write_text("PACKAGE-V3", encoding="utf-8")
                    await run(runtime, "session", "overwritten", initial="PACKAGE-V3", absent=("PACKAGE-V1", "PACKAGE-V2"))
                    rules.unlink()
                    await run(runtime, "session", "deleted", absent=("PACKAGE-V1", "PACKAGE-V2", "PACKAGE-V3"))
                    rules.write_text("PACKAGE-V4", encoding="utf-8")
                    await run(runtime, "session", "recreated", initial="PACKAGE-V4", absent=("PACKAGE-V1", "PACKAGE-V2", "PACKAGE-V3"))
                    current_rule = "PACKAGE-V4"
                target = "other/file.txt"
                await run(runtime, "session", "unknown", initial=current_rule, discover="OTHER-RULE")
                if full_mutation_path:
                    row = await persistence.session_turns.get_session("session")
                    assert await runtime.sessions.fork_turn_session(
                        "session", "fork", principal=runtime.default_principal,
                        expected_revision=row["revision"], idempotency_key="fork",
                    )
                    await run(runtime, "fork", "fork", discover="OTHER-RULE", absent=(current_rule,))
                    await runtime.agents.get("reader").create_session("isolated")
                    await run(runtime, "isolated", "isolated", discover="OTHER-RULE", absent=(current_rule,))
        if native:
            async with Persistence.open("scope-memory", storage=storage) as reopened:
                stored = await reopened.session_turns.get_session("session")
                assert stored["config_json"]["instruction_scopes"] == ["other", "pkg"]
                if split_selects_step_only:
                    # The split Step profile explicitly excludes the workspace
                    # capability. Its rejected admission cannot change learned scopes.
                    async def model_without_tools(_messages, _info):
                        pytest.fail("Unsupported split Step workspace reached the provider")

                    configured = workspace_features("scope-memory", workspace=Workspace.load(workspace_root))
                    configured.capability(StepPersistenceCapability(), scope="runtime")
                    configured.agent("reader", model="default", allow_tools=(), allow_subagents=())
                    model_features(_Models(_UsageFunctionModel(model_without_tools)), configured)
                    async with Runtime.open(persistence=reopened, features=configured, **auth) as runtime:
                        with pytest.raises(AIError) as rejected:
                            await runtime.sessions.start(
                                "session", "split Step reopen", agent="reader", principal=runtime.default_principal,
                                idempotency_key="split-step-reopen",
                            )
                        assert rejected.value.code is ErrorCode.STORAGE_CAPABILITY_MISSING
                    assert await reopened.session_turns.get_session("session") == stored


@pytest.mark.asyncio
async def test_session_remembers_initial_cwd_scopes_and_refresh_failure_is_not_cached(tmp_path: Path) -> None:
    nested = tmp_path / "pkg"
    nested.mkdir()
    rules = nested / "AGENTS.md"
    rules.write_text("CWD-RULE", encoding="utf-8")
    seen = []

    async def model(messages, info):
        del messages
        seen.append(info.instructions)
        assert "CWD-RULE" in (info.instructions or "")
        return ModelResponse(parts=[TextPart("done")])

    features = workspace_features("cwd-scopes", workspace=Workspace.load(tmp_path))
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.agent("reader", model="default", allow_tools=())
    async with Persistence.open("cwd-scopes", storage=PersistenceStorage.in_memory()) as persistence:
        async with Runtime.open(persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), features)) as runtime:
            session = await runtime.agents.get("reader").create_session("session", cwd="pkg")
            assert (await session.run("initial", timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
            current = await persistence.conversation.sessions.get("session", tenant_id=runtime.tenant_id)
            assert current is not None
            await session.update(expected_revision=current.revision, metadata={}, cwd=".")
            rules.write_bytes(b"\xff")
            with pytest.raises(AIError) as failed:
                await session.start("invalid")
            assert failed.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID
            assert len(seen) == 1
            rules.write_text("CWD-RULE", encoding="utf-8")
            assert (await session.run("restored", timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
            assert len(seen) == 2


def _session() -> SessionRecord:
    now = datetime.now(timezone.utc)
    return SessionRecord("session", "owner", SessionStatus.OPEN, 0, None, {}, now, now, None, None, "reader")


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite", "sql"))
async def test_session_scope_union_is_transactional_and_requires_execution_owner(
    tmp_path: Path, backend: str,
) -> None:
    async with _storage(tmp_path, backend) as storage:
        async with Persistence.open("scope-union", storage=storage, tenant_id="tenant") as persistence:
            if backend != "memory":
                from linktools.ai.persistence.turns import SessionInstructionFrontier, SessionTurnKey
                from .test_session_turn_sql import accept

                turns = persistence.session_turns
                await turns.create("chat", config={})
                assert await accept(turns, "active")
                key = SessionTurnKey("chat", "active")
                frontier = SessionInstructionFrontier(RepositoryInstructions((
                    RepositoryInstructionDocument("agents:pkg/AGENTS.md", "pkg", "scope only"),
                )))
                mutation = type(turns._context).run_mutation

                async def simulate_rollback(context, write, *, domain, **options):
                    if context is not turns._context:
                        return await mutation(context, write, domain=domain, **options)
                    async def rollback(connection):
                        await write(connection)
                        raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
                    return await mutation(context, rollback, domain=domain, **options)

                async def remember(scopes, *, generation=1):
                    await turns.append_instruction_frontier(key, generation=generation, frontier=frontier,
                                                            max_bytes=4096, instruction_scopes=scopes)

                with pytest.MonkeyPatch.context() as patch:
                    patch.setattr(type(turns._context), "run_mutation", simulate_rollback)
                    with pytest.raises(AIError) as failed:
                        await remember(("pkg",))
                    assert failed.value.code is ErrorCode.STORAGE_UNAVAILABLE
                assert not (await turns.get_session("chat"))["config_json"]
                assert "instruction_frontier" not in (await turns.get_turn(key))["request_json"]
                await remember(("pkg",))
                with pytest.raises(AIError) as stale:
                    await remember(("forbidden", "pkg"), generation=2)
                assert stale.value.code is ErrorCode.STORAGE_CONFLICT
                assert (await turns.get_session("chat"))["config_json"]["instruction_scopes"] == ["pkg"]
                assert not await turns.update_session("chat", expected_revision=0, config={})
                await remember(("other", "pkg", "third"))
                assert (await turns.get_session("chat"))["config_json"]["instruction_scopes"] == ["other", "pkg", "third"]
                return
            sessions = persistence.conversation.sessions
            await sessions.create(_session())
            await sessions.admit_execution("session", tenant_id="tenant", execution_id="active", expected=None)

            async def remember(transaction, scopes=("pkg",), execution_id="active"):
                return await sessions.remember_repository_instruction_scopes_in_transaction(
                    transaction, "session", tenant_id="tenant", execution_id=execution_id, scopes=scopes,
                )

            async def rollback(transaction):
                await remember(transaction)
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

            with pytest.raises(AIError) as interrupted:
                await sessions.state_store.mutate(rollback)
            assert interrupted.value.code is ErrorCode.STORAGE_UNAVAILABLE
            stored = await sessions.get("session", tenant_id="tenant")
            assert stored is not None and stored.repository_instruction_scopes == ()
            await sessions.state_store.mutate(remember)
            scopes = (("other", "pkg"), ("third", "pkg"))
            results = await asyncio.gather(*(
                sessions.state_store.mutate(lambda tx, value=value: remember(tx, value))
                for value in scopes
            ), return_exceptions=True)
            for value, result in zip(scopes, results):
                if isinstance(result, BaseException):
                    assert isinstance(result, AIError) and result.code is ErrorCode.STORAGE_CONFLICT
                    await sessions.state_store.mutate(lambda tx: remember(tx, value))
            stored = await sessions.get("session", tenant_id="tenant")
            assert stored is not None and stored.repository_instruction_scopes == ("other", "pkg", "third")
            assert decode_domain(encode_domain(stored), SessionRecord) == stored
            with pytest.raises(AIError) as wrong_owner:
                await sessions.state_store.mutate(lambda tx: remember(tx, ("forbidden",), "stale"))
            assert wrong_owner.value.code is ErrorCode.SESSION_BUSY
            with pytest.raises(AIError) as stale:
                await sessions.compare_and_swap(
                    "session", tenant_id="tenant", expected_revision=0,
                    next_record=replace(_session(), revision=1),
                )
            assert stale.value.code is ErrorCode.SESSION_REVISION_CONFLICT
            assert (await sessions.get("session", tenant_id="tenant")).repository_instruction_scopes == ("other", "pkg", "third")


@pytest.mark.parametrize("scopes", (("../outside",), ("pkg", "pkg"), ("pkg", "other"), ["pkg"]))
def test_session_scope_wire_rejects_noncanonical_values(scopes) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(_session(), repository_instruction_scopes=scopes)


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_scope", (False, True))
async def test_session_execution_recovery_keeps_pinned_instructions_and_permission_wait(tmp_path: Path, initial_scope: bool) -> None:
    nested = tmp_path / "pkg"
    nested.mkdir()
    rules = (tmp_path if initial_scope else nested) / "AGENTS.md"
    rules.write_text("PINNED-RULE", encoding="utf-8")
    (nested / "file.txt").write_text("approved contents", encoding="utf-8")
    requests = 0
    pause_requests = 1

    async def model(messages, info):
        nonlocal requests
        requests += 1
        instructions = info.instructions or ""
        if initial_scope:
            assert "PINNED-RULE" in instructions and "LATER-RULE" not in instructions
        elif requests == 1:
            assert "PINNED-RULE" not in instructions
        else:
            assert "LATER-RULE" in instructions and "PINNED-RULE" not in instructions
        if requests == pause_requests + 1:
            assert "approved contents" in str(messages[-1])
            assert not any(is_tool_retry(part) for part in messages[-1].parts)
            return ModelResponse(parts=[TextPart("done")])
        assert requests <= pause_requests
        return ModelResponse(parts=[ToolCallPart("read_file", {"path": "pkg/file.txt"}, f"read-{requests}")])

    workspace = Workspace.load(tmp_path, policy=WorkspacePolicy(
        tool_permissions=ToolPermissionPolicy(default="ask"),
    ))
    features = workspace_features("recover-scopes", workspace=workspace)
    features.capability(SessionCapability(checkpoints=True), scope="runtime")
    features.agent("reader", model="default", allow_tools=("read_file",), tool_retries=0, output_retries=0)
    features = model_features(_Models(_UsageFunctionModel(model)), features)
    storage = PersistenceStorage.sqlite(tmp_path / "runtime.db")
    async with Persistence.open("recover-scopes", storage=storage) as persistence:
        principal = Principal("operator", persistence.tenant_id)
        async with Runtime.open(
            persistence=persistence, features=features, authorization=_HostAuthorization(),
        ) as runtime:
            await runtime.agents.get("reader").create_session("session", principal=principal)
            turn = await runtime.sessions.start(
                "session", "read", agent="reader", principal=principal, idempotency_key="read-once",
            )
            assert (await asyncio.wait_for(turn.wait(), timeout=10)).status is ExecutionStatus.WAITING_DEFERRED
            [waiting] = await runtime.approvals.list(turn.key, principal=principal)
            assert requests == pause_requests
        rules.write_text("LATER-RULE", encoding="utf-8")
        async with Runtime.open(
            persistence=persistence, features=features, authorization=_HostAuthorization(),
        ) as recovered:
            [approval] = await recovered.approvals.list(turn.key, principal=principal)
            assert approval.approval_id == waiting.approval_id
            assert requests == pause_requests
            attached = await recovered.sessions.start(
                "session", "read", agent="reader", principal=principal, idempotency_key="read-once",
            )
            assert attached.key == turn.key
            await recovered.approvals.decide(turn.key, ApprovalDecisionRequest(
                principal, approval.approval_id, "approve-exact", ApprovalDecision.APPROVE,
            ))
            assert (await attached.wait_final(timeout_seconds=10)).status is ExecutionStatus.SUCCEEDED
            assert requests == pause_requests + 1
            assert await recovered.approvals.list(turn.key, principal=principal) == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("scope", "pin"))
async def test_discovery_scope_commit_precedes_execution_pin_across_storage_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    overlay = RepositoryInstructions((RepositoryInstructionDocument("agents:pkg/AGENTS.md", "pkg", "rule"),))
    storage = PersistenceStorage.from_plan(PersistencePlan(
        execution=StorageRoute.sqlite(tmp_path / "execution.db"),
        conversation=StorageRoute.memory(),
    ))
    async with Persistence.open("scope-pin-order", storage=storage, tenant_id="tenant") as persistence:
        sessions = persistence.conversation.sessions
        assert sessions is not None
        assert sessions.state_store.storage_group is not persistence.execution.executions.state_store.storage_group
        await sessions.create(_session())
        await sessions.admit_execution("session", tenant_id="tenant", execution_id="active", expected=None)
        pin_attempts = []

        async def promote(execution_id):
            assert execution_id == "active"

        async def store_pin(execution, instructions):
            del execution, instructions
            pin_attempts.append(True)
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

        async def fail_scope(*args, **kwargs):
            del args, kwargs
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

        if failure == "scope":
            monkeypatch.setattr(sessions, "remember_repository_instruction_scopes_in_transaction", fail_scope)
        port = SimpleNamespace(
            _tenant_id="tenant", _legacy_sessions=lambda: sessions, _turn_owners={},
            promote_pending_execution=promote, _store_repository_instructions=store_pin,
        )
        with pytest.raises(AIError) as interrupted:
            await LocalExecutionBackend.commit_repository_instruction_overlay(
                port,
                SimpleNamespace(execution_id="active", session_id="session",
                                recovery=ExecutionRecoveryDetail(agent_run_id="run")),
                overlay,
            )
        assert interrupted.value.code is ErrorCode.STORAGE_UNAVAILABLE
        stored = await sessions.get("session", tenant_id="tenant")
        assert stored is not None
        assert stored.repository_instruction_scopes == (() if failure == "scope" else ("pkg",))
        assert bool(pin_attempts) is (failure == "pin")


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ("files", "inline", "both"))
@pytest.mark.parametrize(("native", "session_backed"), ((False, False), (False, True), (True, True)))
async def test_attachment_scopes_are_in_first_request_and_sources_are_deduplicated(
    tmp_path: Path, entry: str, session_backed: bool, native: bool,
) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "AGENTS.md").write_text("ATTACHMENT-AGENTS", encoding="utf-8")
    (package / "file.txt").write_text("attachment contents", encoding="utf-8")
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    try:
        await store.put(AssetKey("rule", "attachment.md"), b"---\nscope: pkg\n---\nATTACHMENT-RULE")
        requests = []

        async def model(messages, info):
            del messages
            instructions = info.instructions or ""
            assert instructions.count("ATTACHMENT-AGENTS") == 1
            assert instructions.count("ATTACHMENT-RULE") == 1
            requests.append(instructions)
            return ModelResponse(parts=[TextPart("done")])

        features = workspace_features("attachment-scopes", assets=store, workspace=Workspace.load(
            tmp_path, policy=WorkspacePolicy(max_repository_instruction_documents=2),
        ))
        if not native:
            features.capability(ExecutionCapability(retain_history=True), scope="runtime")
        features.agent("reader", model="default", allow_tools=(), allow_subagents=())
        features = model_features(_Models(_UsageFunctionModel(model)), features)
        storage = PersistenceStorage.from_root(tmp_path / "state") if native else PersistenceStorage.in_memory()
        auth = {"authorization": _HostAuthorization()} if native else {}
        async with Persistence.open("attachment-scopes", storage=storage) as persistence:
            async with Runtime.open(persistence=persistence, features=features, **auth) as runtime:
                agent = runtime.agents.get("reader")
                target = await agent.create_session("session") if session_backed else agent
                prompt = ("inspect", WorkspaceFileInput("pkg/file.txt")) if entry != "files" else "inspect"
                files = ("pkg/file.txt",) if entry != "inline" else ()
                if native:
                    handle = await runtime.sessions.start(
                        "session", prompt, agent=agent, principal=runtime.default_principal,
                        idempotency_key="attachment", files=files,
                    )
                    result = await handle.wait_final(timeout_seconds=10)
                    session = await persistence.session_turns.get_session("session")
                    assert session["config_json"]["instruction_scopes"] == ["pkg"]
                else:
                    result = (await target.run(prompt, files=files, timeout_seconds=10)).result
                    if session_backed:
                        session = await persistence.conversation.sessions.get("session", tenant_id=runtime.tenant_id)
                        assert session is not None and session.repository_instruction_scopes == ("pkg",)
                assert result.status is ExecutionStatus.SUCCEEDED
                assert len(requests) == 1
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ("files", "remembered"))
async def test_initial_scope_union_is_validated_before_session_scope_write(
    tmp_path: Path, entry: str,
) -> None:
    (tmp_path / "AGENTS.md").write_text("root", encoding="utf-8")
    for name in ("pkg", "other"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "AGENTS.md").write_text(name, encoding="utf-8")
        (directory / "file.txt").write_text(name, encoding="utf-8")

    async def model(messages, info):
        del messages, info
        raise AssertionError("over-limit instruction union must not reach the model")

    features = workspace_features("initial-limits", workspace=Workspace.load(
        tmp_path, policy=WorkspacePolicy(max_repository_instruction_documents=2),
    ))
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.agent("reader", model="default", allow_tools=())
    features = model_features(_Models(_UsageFunctionModel(model)), features)
    async with Persistence.open("initial-limits", storage=PersistenceStorage.in_memory()) as persistence:
        async with Runtime.open(persistence=persistence, features=features) as runtime:
            session = await runtime.agents.get("reader").create_session("session")
            scopes = ("other", "pkg") if entry == "remembered" else ()
            sessions = persistence.conversation.sessions
            original = await sessions.get("session", tenant_id=runtime.tenant_id)
            assert original is not None
            if scopes:
                original = await sessions.compare_and_swap(
                    "session", tenant_id=runtime.tenant_id, expected_revision=original.revision,
                    next_record=replace(original, revision=original.revision + 1, repository_instruction_scopes=scopes),
                )
            files = ("pkg/file.txt", "other/file.txt") if entry == "files" else ()
            with pytest.raises(AIError) as raised:
                await session.start("inspect", files=files)
            assert raised.value.code is ErrorCode.PROMPT_TOO_LARGE
            stored = await sessions.get("session", tenant_id=runtime.tenant_id)
            assert stored is not None and stored.repository_instruction_scopes == scopes
            assert stored.revision == original.revision and stored.active_execution_id is None


@pytest.mark.asyncio
async def test_subagent_attachments_extend_but_do_not_refresh_parent_pin_or_session(tmp_path: Path) -> None:
    root_rule = tmp_path / "AGENTS.md"
    root_rule.write_text("PARENT-PINNED", encoding="utf-8")
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "AGENTS.md").write_text("CHILD-ATTACHMENT-RULE", encoding="utf-8")
    (package / "file.txt").write_text("child attachment", encoding="utf-8")
    child_requests = []
    parent_requests = 0

    async def model(messages, info):
        nonlocal parent_requests
        instructions = info.instructions or ""
        assert "PARENT-PINNED" in instructions
        assert "MUTATED-ROOT" not in instructions
        if any(tool.name == "delegate_task" for tool in info.function_tools):
            parent_requests += 1
            assert "CHILD-ATTACHMENT-RULE" not in instructions
            if parent_requests == 1:
                root_rule.write_text("MUTATED-ROOT", encoding="utf-8")
                return ModelResponse(parts=[ToolCallPart("delegate_task", {
                    "subagent_id": "child", "task": "inspect", "files": ["pkg/file.txt"],
                }, "delegate")])
            return ModelResponse(parts=[TextPart("done")])
        child_requests.append(instructions)
        assert "CHILD-ATTACHMENT-RULE" in instructions
        return ModelResponse(parts=[TextPart("child done")])

    features = workspace_features("child-attachment", workspace=Workspace.load(tmp_path))
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.agent("reader", model="default", allow_tools=(), allow_subagents=("child",))
    features.agent("child", model="default", allow_tools=(), allow_subagents=())
    features = model_features(_Models(_UsageFunctionModel(model)), features)
    async with Persistence.open("child-attachment", storage=PersistenceStorage.in_memory()) as persistence:
        async with Runtime.open(persistence=persistence, features=features) as runtime:
            session = await runtime.agents.get("reader").create_session("session")
            result = (await session.run("delegate", timeout_seconds=10)).result
            assert result.status is ExecutionStatus.SUCCEEDED
            assert len(child_requests) == 1 and parent_requests == 2
            stored = await persistence.conversation.sessions.get("session", tenant_id=runtime.tenant_id)
            assert stored is not None and stored.repository_instruction_scopes == (".",)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("native", "custom"))
async def test_native_and_custom_approvals_restore_validated_arguments(
    tmp_path: Path, kind: str,
) -> None:
    from pydantic_ai.exceptions import ApprovalRequired
    from linktools.ai.capability import AgentContext, CapabilityContribution

    (tmp_path / "AGENTS.md").write_text("APPROVAL-RULE", encoding="utf-8")
    calls = []
    requested = False
    requests = 0

    async def checked(context: AgentContext[object], value: int, suffix: str = "default") -> str:
        nonlocal requested
        del context
        if kind == "custom" and not requested:
            requested = True
            raise ApprovalRequired(metadata={"source": "custom"})
        calls.append((value, suffix))
        return f"{value}:{suffix}"

    async def model(messages, info):
        nonlocal requests
        requests += 1
        assert "APPROVAL-RULE" in (info.instructions or "")
        if requests == 1:
            # Native approval freezes the raw call; SDK validation fills the
            # default and converts this string on approved replay.
            return ModelResponse(parts=[ToolCallPart("checked", {"value": "7"}, "original")])
        assert requests == 2
        assert "7:default" in str(messages[-1])
        assert not any(is_tool_retry(part) for part in messages[-1].parts)
        return ModelResponse(parts=[TextPart("done")])

    features = workspace_features("native-approval", workspace=Workspace.load(tmp_path))
    source = features.tool(checked, effect_policy="none")
    source.requires_approval = kind == "native"
    features.agent("reader", model="default", allow_tools=("checked",), tool_retries=0, output_retries=0)
    model_features(_Models(_UsageFunctionModel(model)), features)
    features._contributions = [
        CapabilityContribution.from_opaque("tool", "checked", source)
        if item.kind == "tool" and item.id == "checked" else item
        for item in features._contributions
    ]
    async with Persistence.open("native-approval", storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as persistence:
        async with Runtime.open(persistence=persistence, features=features) as runtime:
            execution = await runtime.agents.get("reader").start("check")
            tenant = runtime.tenant_id
            principal_id = runtime.default_principal.principal_id
            await _wait_for_committed(
                lambda: persistence.execution.executions.get(execution.execution_id, tenant_id=tenant),
                lambda record: record is not None and record.status is ExecutionStatus.WAITING_DEFERRED,
            )
            waiting = await persistence.execution.executions.get(execution.execution_id, tenant_id=tenant)
            assert waiting is not None
            assert not waiting.conversation and not waiting.restart_checkpoints and not waiting.durable_trace
            approvals = await persistence.checkpoint.approvals.list_pending(execution.execution_id, tenant_id=tenant)
            assert len(approvals) == 1 and calls == [] and requests == 1
        await persistence.checkpoint.approvals.decide(
            approvals[0].approval_id, tenant_id=tenant, expected_status=ApprovalStatus.PENDING,
            idempotency_key="c" * 64, decision=ApprovalDecision.APPROVE,
            principal_id=principal_id, decision_digest="d" * 64,
            decided_at=datetime.now(timezone.utc),
            resolution_metadata={"operator_note": "approved exact tool call"},
        )
        async with Runtime.open(persistence=persistence, features=features, auto_recover=True) as runtime:
            finished = await runtime.executions.wait(
                execution.execution_id, principal=runtime.default_principal, timeout_seconds=10,
            )
            assert finished.result.status is ExecutionStatus.SUCCEEDED
            assert requests == 2 and calls == [(7, "default")]
            assert await persistence.checkpoint.approvals.list_pending(execution.execution_id, tenant_id=tenant) == ()


@pytest.mark.asyncio
async def test_concurrent_runtime_initial_scope_discovery_preserves_same_request_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import AsyncExitStack
    from linktools.ai.workspace import WorkspaceInstructionResolver

    (tmp_path / "AGENTS.md").write_text("INITIAL-RULE", encoding="utf-8")
    cache = InMemoryCacheBackend(max_bytes=2**20)
    both_resolving = asyncio.Event()
    arrivals = 0
    original_resolve = WorkspaceInstructionResolver.resolve

    async def resolve(resolver, path=".", *, exclude_sources=frozenset(), within=None):
        nonlocal arrivals
        if path == "." and arrivals < 2:
            arrivals += 1
            if arrivals == 2:
                both_resolving.set()
            await asyncio.wait_for(both_resolving.wait(), timeout=5)
        return await original_resolve(resolver, path, exclude_sources=exclude_sources, within=within)

    monkeypatch.setattr(WorkspaceInstructionResolver, "resolve", resolve)
    requests = []

    async def model(messages, info):
        del messages
        assert "INITIAL-RULE" in (info.instructions or "")
        requests.append(info.instructions)
        return ModelResponse(parts=[TextPart("done")])

    features = workspace_features("scope-admission", workspace=Workspace.load(tmp_path))
    features.capability(SessionCapability(checkpoints=True), scope="runtime")
    features.agent("reader", model="default", allow_tools=())
    features = model_features(_Models(_UsageFunctionModel(model)), features).borrow()
    async with AsyncExitStack() as stack:
        runtimes = []
        for _ in range(2):
            persistence = await stack.enter_async_context(Persistence.open(
                "scope-admission", storage=PersistenceStorage.sqlite(tmp_path / "state.db", coordination=cache),
            ))
            runtimes.append(await stack.enter_async_context(Runtime.open(
                persistence=persistence, features=features, authorization=_HostAuthorization(),
            )))
        principal = runtimes[0].default_principal
        await runtimes[0].agents.get("reader").create_session("session", principal=principal)
        handles = await asyncio.gather(*(
            runtime.sessions.start("session", "same", agent="reader", principal=principal, idempotency_key="same-key")
            for runtime in runtimes
        ))
        assert handles[0].key == handles[1].key
        # This fixture has two runtimes without a shared observer carrier;
        # observe durable settlement rather than subscribe from the losing host.
        stored = await _wait_for_committed(
            lambda: persistence.session_turns.get_turn(handles[0].key),
            lambda row: row is not None and row["state"] == "succeeded",
        )
        assert arrivals == 2 and len(requests) == 1
        instructions = RepositoryInstructions.from_payload(stored["request_json"]["repository_instructions"])
        assert tuple(document.content for document in instructions.documents) == ("INITIAL-RULE",)


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", (False, True))
@pytest.mark.parametrize("imported", (False, True))
async def test_accepted_context_owns_instruction_pin_before_model_dispatch(
    tmp_path: Path, terminal: bool, imported: bool,
) -> None:
    from linktools.ai.execution import ExecutionInputContext, ExecutionRecord
    from linktools.ai.persistence.codec import (
        _decode_enveloped_domain, _encode_persisted_domain, encode_envelope,
    )

    (tmp_path / "AGENTS.md").write_text("LIVE-INSTRUCTION", encoding="utf-8")
    expected = "CAPTURED-INSTRUCTION" if imported else "LIVE-INSTRUCTION"
    context = ExecutionInputContext.from_messages((), repository_instructions=RepositoryInstructions((
        RepositoryInstructionDocument("agents:AGENTS.md", ".", "CAPTURED-INSTRUCTION"),
    )).to_payload()) if imported else None
    ready = asyncio.Event()
    execution_id = ""
    observed = []

    async def model(messages, info):
        await ready.wait()
        assert expected in (info.instructions or "")
        if imported:
            assert "LIVE-INSTRUCTION" not in (info.instructions or "")
        backend = runtime._execution_service.runtime_backend()
        record = await backend.load_execution(execution_id, tenant_id=runtime.tenant_id)
        assert record is not None and record.input_context is not None
        assert record.repository_instructions is None
        accepted = await backend.load_input_context(record.input_context)
        assert accepted is not None and accepted.repository_instructions is not None
        assert tuple(document.content for document in RepositoryInstructions.from_payload(
            accepted.repository_instructions,
        ).documents) == (expected,)
        restored = _decode_enveloped_domain(encode_envelope({
            "type": "execution_record", "payload": _encode_persisted_domain(record),
        }), ExecutionRecord)
        assert restored == record
        observed.append(record)
        return ModelResponse(parts=[TextPart("done")])

    features = workspace_features("instruction-owner", workspace=Workspace.load(tmp_path))
    features.capability(ExecutionCapability(
        retain_history=True, commit=CommitPolicy.terminal() if terminal else CommitPolicy.sync(),
    ), scope="runtime")
    features.agent("reader", model="default", allow_tools=())
    storage = PersistenceStorage.in_memory(
        coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024) if terminal else None,
    )
    async with Persistence.open("instruction-owner", storage=storage) as persistence:
        async with Runtime.open(
            persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), features),
        ) as runtime:
            agent = runtime.agents.get("reader")
            if imported:
                handle = await agent.start("question", input_context=context)
            else:
                session = await agent.create_session("session")
                handle = await session.start("question")
            execution_id = handle.execution_id
            ready.set()
            assert (await handle.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
            assert len(observed) == 1
            stored = await persistence.execution.executions.get(execution_id, tenant_id=runtime.tenant_id)
            assert stored is not None and stored.repository_instructions is None
            accepted = await runtime._execution_service.runtime_backend().load_input_context(stored.input_context)
            assert accepted is not None and accepted.repository_instructions is not None
            assert tuple(document.content for document in RepositoryInstructions.from_payload(
                accepted.repository_instructions,
            ).documents) == (expected,)
