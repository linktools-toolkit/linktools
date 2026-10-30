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
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.asset import AssetKey, AssetStore, InMemoryAssetBackend
from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import ApprovalDecision, ApprovalStatus, ExecutionStatus, SessionStatus, WorkspaceFileInput
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.persistence import Persistence, PersistencePlan, PersistenceStorage
from linktools.ai.persistence.codec import decode_domain, encode_domain
from linktools.ai.runtime import Runtime
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.persistence.contracts import RepositoryInstructionBarrier
from linktools.ai.spec import RepositoryInstructionDocument, RepositoryInstructions
from linktools.ai.storage import StorageOverlay, StorageRoute
from linktools.ai.session import SessionRecord
from linktools.ai.workspace import ToolPermissionPolicy, Workspace, WorkspacePolicy

from ._runtime_test_helpers import (
    _Models,
    _UsageFunctionModel,
    _wait_for_committed,
    is_tool_retry,
    model_features,
    workspace_features,
)


@asynccontextmanager
async def _storage(root: Path, backend: str) -> AsyncIterator[PersistenceStorage]:
    engine = None
    if backend == "memory":
        storage = PersistenceStorage.in_memory()
    elif backend == "filesystem":
        storage = PersistenceStorage.filesystem(root / "state")
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
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite", "sql", "split"))
async def test_session_reloads_known_scopes_before_first_model_request(
    tmp_path: Path, backend: str,
) -> None:
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
        elif attempts == 2 and discovery_rule is not None:
            assert discovery_rule in instructions
            assert any(is_tool_retry(part) for part in messages[-1].parts)
        else:
            assert not any(is_tool_retry(part) for part in messages[-1].parts)
            assert "contents" in str(messages[-1])
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart(
            "read_file", {"path": target}, f"{turn}-{attempts}",
        )])

    features = workspace_features("scope-memory", workspace=Workspace.load(workspace_root))
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.agent("reader", model="default", allow_tools=("read_file",))
    features = model_features(_Models(_UsageFunctionModel(model)), features)

    async def run(session, name: str, *, initial: str | None = None,
                  discover: str | None = None, absent: tuple[str, ...] = ()) -> None:
        nonlocal turn, attempts, initial_rule, discovery_rule, forbidden_rules
        turn, attempts = name, 0
        initial_rule, discovery_rule, forbidden_rules = initial, discover, absent
        result = (await session.run(name, timeout_seconds=10)).result
        assert result.status is ExecutionStatus.SUCCEEDED
        assert attempts == (3 if discover is not None else 2)

    async with _storage(tmp_path, backend) as storage:
        async with Persistence.open("scope-memory", storage=storage) as persistence:
            async with Runtime.attach(persistence=persistence, features=features) as runtime:
                session = await runtime.agents.get("reader").create_session("session")
                await run(session, "learn", discover="PACKAGE-V1")
                stored = await persistence.conversation.sessions.get("session", tenant_id=runtime.tenant_id)
                assert stored is not None and stored.repository_instruction_scopes == ("pkg",)
                assert "PACKAGE-V1" not in str(encode_domain(stored))

            # A new Runtime must consume the durable Session scope set, not a local cache.
            async with Runtime.attach(persistence=persistence, features=features) as runtime:
                session = runtime.agents.get("reader").session("session")
                await run(session, "same", initial="PACKAGE-V1")
                rules.write_text("PACKAGE-V2", encoding="utf-8")
                await run(session, "changed", initial="PACKAGE-V2", absent=("PACKAGE-V1",))
                rules.unlink()
                await run(session, "deleted", absent=("PACKAGE-V1", "PACKAGE-V2"))
                rules.write_text("PACKAGE-V3", encoding="utf-8")
                await run(session, "recreated", initial="PACKAGE-V3", absent=("PACKAGE-V1", "PACKAGE-V2"))
                target = "other/file.txt"
                await run(session, "unknown", initial="PACKAGE-V3", discover="OTHER-RULE")
                fork = await session.fork("fork")
                await run(fork, "fork", discover="OTHER-RULE", absent=("PACKAGE-V3",))
                isolated = await runtime.agents.get("reader").create_session("isolated")
                await run(isolated, "isolated", discover="OTHER-RULE", absent=("PACKAGE-V3",))
        if backend != "memory":
            async with Persistence.open("scope-memory", storage=storage) as reopened:
                stored = await reopened.conversation.sessions.get("session", tenant_id=reopened.tenant_id)
                assert stored is not None and stored.repository_instruction_scopes == ("other", "pkg")


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
        async with Runtime.attach(persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), features)) as runtime:
            session = await runtime.agents.get("reader").create_session("session", cwd="pkg")
            assert (await session.run("initial", timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
            current = await runtime.sessions.get("session", principal=runtime.default_principal)
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
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite", "sql"))
async def test_session_scope_union_is_transactional_and_requires_execution_owner(
    tmp_path: Path, backend: str,
) -> None:
    async with _storage(tmp_path, backend) as storage:
        async with Persistence.open("scope-union", storage=storage, tenant_id="tenant") as persistence:
            sessions = persistence.conversation.sessions
            await sessions.create(_session())
            await sessions.admit_execution("session", tenant_id="tenant", execution_id="active", expected=None)

            async def remember(transaction, scopes=("pkg",), execution_id="active"):
                return await sessions.remember_repository_instruction_scopes_in_transaction(
                    transaction, "session", tenant_id="tenant", execution_id=execution_id, scopes=scopes,
                )

            async def rollback(transaction):
                await remember(transaction)
                raise RuntimeError("scope transaction interrupted")

            with pytest.raises(RuntimeError, match="scope transaction interrupted"):
                await sessions.state_store.mutate(rollback)
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
    pause_requests = 1 if initial_scope else 2

    async def model(messages, info):
        nonlocal requests
        requests += 1
        instructions = info.instructions or ""
        assert "LATER-RULE" not in instructions
        if requests == 1 and not initial_scope:
            assert "PINNED-RULE" not in instructions
        else:
            assert "PINNED-RULE" in instructions
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
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.agent("reader", model="default", allow_tools=("read_file",))
    features = model_features(_Models(_UsageFunctionModel(model)), features)
    storage = PersistenceStorage.sqlite(tmp_path / "runtime.db")
    async with Persistence.open("recover-scopes", storage=storage) as persistence:
        async with Runtime.attach(persistence=persistence, features=features) as runtime:
            session = await runtime.agents.get("reader").create_session("session")
            execution = await session.start("read")
            tenant = runtime.tenant_id
            await _wait_for_committed(
                lambda: persistence.execution.executions.get(execution.execution_id, tenant_id=tenant),
                lambda record: record is not None and record.status is ExecutionStatus.WAITING_DEFERRED,
            )
            approvals = await persistence.recovery.approvals.list_pending(execution.execution_id, tenant_id=tenant)
            assert len(approvals) == 1 and requests == pause_requests
            checkpoint = await persistence.recovery.checkpoints.get(execution.execution_id, tenant_id=tenant)
            assert checkpoint is not None and checkpoint.pending_tools is not None
            assert len(checkpoint.pending_tools.approvals[0].metadata["linktools"]["repository_instruction_frontier"]) == 64
            rules.write_text("LATER-RULE", encoding="utf-8")
            await persistence.recovery.approvals.decide(
                approvals[0].approval_id, tenant_id=tenant, expected_status=ApprovalStatus.PENDING,
                idempotency_key_digest="a" * 64, decision=ApprovalDecision.APPROVE,
                principal_id=runtime.default_principal.principal_id, decision_digest="b" * 64,
                decided_at=datetime.now(timezone.utc),
                resolution_metadata={"linktools": {"repository_instruction_frontier": "0" * 64}},
            )
        async with Runtime.attach(persistence=persistence, features=features) as runtime:
            finished = await runtime.executions.wait(
                execution.execution_id, principal=runtime.default_principal, timeout_seconds=10,
            )
            assert finished.result.status is ExecutionStatus.SUCCEEDED
            assert requests == pause_requests + 1
            assert await persistence.recovery.approvals.list_pending(execution.execution_id, tenant_id=tenant) == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("scope", "pin"))
async def test_discovery_scope_commit_precedes_execution_pin_across_storage_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    overlay = RepositoryInstructions((RepositoryInstructionDocument("agents:pkg/AGENTS.md", "pkg", "rule"),))
    async with _storage(tmp_path, "split") as storage:
        async with Persistence.open("scope-pin-order", storage=storage, tenant_id="tenant") as persistence:
            sessions = persistence.conversation.sessions
            await sessions.create(_session())
            await sessions.admit_execution("session", tenant_id="tenant", execution_id="active", expected=None)
            pin_attempts = []

            async def promote(execution_id):
                assert execution_id == "active"

            async def store_pin(execution, instructions):
                del execution, instructions
                pin_attempts.append(True)
                raise RuntimeError("pin failed")

            async def fail_scope(*args, **kwargs):
                del args, kwargs
                raise RuntimeError("scope failed")

            if failure == "scope":
                monkeypatch.setattr(sessions, "remember_repository_instruction_scopes_in_transaction", fail_scope)
            port = SimpleNamespace(
                _tenant_id="tenant", _conversation=persistence.conversation,
                promote_pending_execution=promote, _store_repository_instructions=store_pin,
            )
            with pytest.raises(RuntimeError, match=f"{failure} failed"):
                await LocalExecutionBackend.commit_repository_instruction_barrier(
                    port,
                    SimpleNamespace(execution_id="active", session_id="session"),
                    SimpleNamespace(repository_instruction_barriers=()),
                    overlay, RepositoryInstructionBarrier("run", "call", "a" * 64),
                )
            stored = await sessions.get("session", tenant_id="tenant")
            assert stored is not None
            assert stored.repository_instruction_scopes == (() if failure == "scope" else ("pkg",))
            assert bool(pin_attempts) is (failure == "pin")


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ("files", "inline", "both"))
@pytest.mark.parametrize("session_backed", (False, True))
async def test_attachment_scopes_are_in_first_request_and_sources_are_deduplicated(
    tmp_path: Path, entry: str, session_backed: bool,
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
        features.capability(ExecutionCapability(retain_history=True), scope="runtime")
        features.agent("reader", model="default", allow_tools=())
        features = model_features(_Models(_UsageFunctionModel(model)), features)
        async with Persistence.open("attachment-scopes", storage=PersistenceStorage.in_memory()) as persistence:
            async with Runtime.attach(persistence=persistence, features=features) as runtime:
                agent = runtime.agents.get("reader")
                target = await agent.create_session("session") if session_backed else agent
                prompt = ("inspect", WorkspaceFileInput("pkg/file.txt")) if entry != "files" else "inspect"
                files = ("pkg/file.txt",) if entry != "inline" else ()
                result = (await target.run(prompt, files=files, timeout_seconds=10)).result
                assert result.status is ExecutionStatus.SUCCEEDED
                assert len(requests) == 1
                if session_backed:
                    session = await persistence.conversation.sessions.get("session", tenant_id=runtime.tenant_id)
                    assert session is not None and session.repository_instruction_scopes == ("pkg",)
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
        async with Runtime.attach(persistence=persistence, features=features) as runtime:
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
        async with Runtime.attach(persistence=persistence, features=features) as runtime:
            session = await runtime.agents.get("reader").create_session("session")
            result = (await session.run("delegate", timeout_seconds=10)).result
            assert result.status is ExecutionStatus.SUCCEEDED
            assert len(child_requests) == 1 and parent_requests == 2
            stored = await persistence.conversation.sessions.get("session", tenant_id=runtime.tenant_id)
            assert stored is not None and stored.repository_instruction_scopes == (".",)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("native", "custom"))
async def test_native_and_custom_approvals_restore_observed_frontier_with_validated_arguments(
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
    features.agent("reader", model="default", allow_tools=("checked",))
    capture = await model_features(_Models(_UsageFunctionModel(model)), features).capture()
    features = replace(capture, contributions=tuple(
        CapabilityContribution.from_opaque("tool", "checked", source)
        if item.kind == "tool" and item.id == "checked" else item
        for item in capture.contributions
    ))
    async with Persistence.open("native-approval", storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as persistence:
        async with Runtime.attach(persistence=persistence, features=features) as runtime:
            execution = await runtime.agents.get("reader").start("check")
            tenant = runtime.tenant_id
            principal_id = runtime.default_principal.principal_id
            await _wait_for_committed(
                lambda: persistence.execution.executions.get(execution.execution_id, tenant_id=tenant),
                lambda record: record is not None and record.status is ExecutionStatus.WAITING_DEFERRED,
            )
            approvals = await persistence.recovery.approvals.list_pending(execution.execution_id, tenant_id=tenant)
            assert len(approvals) == 1 and calls == [] and requests == 1
        await persistence.recovery.approvals.decide(
            approvals[0].approval_id, tenant_id=tenant, expected_status=ApprovalStatus.PENDING,
            idempotency_key_digest="c" * 64, decision=ApprovalDecision.APPROVE,
            principal_id=principal_id, decision_digest="d" * 64,
            decided_at=datetime.now(timezone.utc),
            resolution_metadata={"linktools": {"repository_instruction_frontier": "0" * 64}},
        )
        async with Runtime.attach(persistence=persistence, features=features) as runtime:
            finished = await runtime.executions.wait(
                execution.execution_id, principal=runtime.default_principal, timeout_seconds=10,
            )
            assert finished.result.status is ExecutionStatus.SUCCEEDED
            assert requests == 2 and calls == [(7, "default")]
            assert await persistence.recovery.approvals.list_pending(execution.execution_id, tenant_id=tenant) == ()


@pytest.mark.asyncio
async def test_concurrent_runtime_initial_scope_discovery_preserves_same_request_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import AsyncExitStack
    from linktools.ai.workspace import WorkspaceInstructionResolver

    (tmp_path / "AGENTS.md").write_text("INITIAL-RULE", encoding="utf-8")
    both_resolving = asyncio.Event()
    arrivals = 0
    original_resolve = WorkspaceInstructionResolver.resolve

    async def resolve(resolver, path=".", *, exclude_sources=frozenset()):
        nonlocal arrivals
        if path == "." and arrivals < 2:
            arrivals += 1
            if arrivals == 2:
                both_resolving.set()
            await asyncio.wait_for(both_resolving.wait(), timeout=5)
        return await original_resolve(resolver, path, exclude_sources=exclude_sources)

    monkeypatch.setattr(WorkspaceInstructionResolver, "resolve", resolve)
    requests = []

    async def model(messages, info):
        del messages
        assert "INITIAL-RULE" in (info.instructions or "")
        requests.append(info.instructions)
        return ModelResponse(parts=[TextPart("done")])

    features = workspace_features("scope-admission", workspace=Workspace.load(tmp_path))
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.agent("reader", model="default", allow_tools=())
    features = await model_features(_Models(_UsageFunctionModel(model)), features).capture()
    async with AsyncExitStack() as stack:
        runtimes = []
        for _ in range(2):
            persistence = await stack.enter_async_context(Persistence.open(
                "scope-admission", storage=PersistenceStorage.sqlite(tmp_path / "state.db"),
            ))
            runtimes.append(await stack.enter_async_context(Runtime.attach(persistence=persistence, features=features)))
        session = await runtimes[0].agents.get("reader").create_session("session")
        handles = await asyncio.gather(
            session.start("same", idempotency_key="same-key"),
            runtimes[1].agents.get("reader").session("session").start("same", idempotency_key="same-key"),
        )
        assert handles[0].execution_id == handles[1].execution_id
        finished = await runtimes[0].executions.wait(
            handles[0].execution_id, principal=runtimes[0].default_principal, timeout_seconds=10,
        )
        assert finished.result.status is ExecutionStatus.SUCCEEDED
        assert len(requests) == 1
        stored = await persistence.conversation.sessions.get("session", tenant_id=runtimes[0].tenant_id)
        assert stored is not None and stored.repository_instruction_scopes == (".",)
