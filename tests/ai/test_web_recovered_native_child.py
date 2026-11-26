#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A reopened Web turn can first-start only its exact untouched child."""

from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from pydantic_ai.messages import BinaryContent, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart

from linktools.ai.capability import ExecutionCapability, FileSystemCapability, StepPersistenceCapability
from linktools.ai.core import ExecutionStatus, IdempotencyStatus, ToolOperationStatus, canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.turns import SessionTurnKey
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.runtime._native_subagent import NativeSubagentAdmission
from linktools.ai.workspace import ToolPermissionPolicy, Workspace, WorkspacePolicy

from ._runtime_test_helpers import _wait_for_committed
from .test_canonical_turn_subagents import _HEADERS, _SESSION, _HostAuthorization, _model


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("recovered", "unknown-start-no-dispatch", "reservation-missing"))
async def test_web_recovery_first_starts_only_the_original_reserved_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str,
) -> None:
    pytest.importorskip("starlette")
    from linktools.ai.web import create_app

    project = tmp_path / "project"
    project.mkdir()
    root_rule = project / "AGENTS.md"
    root_rule.write_text("PINNED-PARENT-RULE", encoding="utf-8")
    child_folder = project / "child"
    child_folder.mkdir()
    child_rule = child_folder / "AGENTS.md"
    child_rule.write_text("PINNED-CHILD-RULE", encoding="utf-8")
    attachment = child_folder / "fixture.txt"
    original_bytes = b"captured before interruption"
    attachment.write_bytes(original_bytes)
    workspace = Workspace.load(project, policy=WorkspacePolicy(tool_permissions=ToolPermissionPolicy(default="allow")))
    calls: list[str] = []
    returned_children: list[str] = []
    invocation = "original-delegate"

    async def parent(messages, info):
        assert "PINNED-PARENT-RULE" in (info.instructions or "")
        assert "PINNED-CHILD-RULE" not in (info.instructions or "")
        results = [part.content for message in messages for part in message.parts
                   if isinstance(part, ToolReturnPart) and part.tool_call_id == invocation]
        if results:
            calls.append("parent continuation")
            [result] = results
            assert result["status"] == "SUCCEEDED" and result["output"] == {"text": "read the retained file"}
            returned_children.append(result["execution_id"])
            return ModelResponse(parts=[TextPart("completed from retained delegation")])
        calls.append("parent dispatch")
        root_rule.write_text("CHANGED-PARENT-RULE", encoding="utf-8")
        return ModelResponse(parts=[ToolCallPart(
            "delegate_task", {"subagent_id": "child", "task": "Read the retained file",
                              "files": [attachment.relative_to(project).as_posix()]},
            tool_call_id=invocation,
        )])

    async def child(messages, info):
        calls.append("child dispatch")
        instructions = info.instructions or ""
        assert "PINNED-PARENT-RULE" in instructions and "PINNED-CHILD-RULE" in instructions
        assert "CHANGED-PARENT-RULE" not in instructions and "CHANGED-CHILD-RULE" not in instructions
        [prompt] = [part for message in messages for part in message.parts if isinstance(part, UserPromptPart)]
        assert [item.data for item in prompt.content if isinstance(item, BinaryContent)] == [original_bytes]
        return ModelResponse(parts=[TextPart("read the retained file")])

    async def stop_before_start(_proof, _commands, _claim):
        # Exact failure point: the original reservation and tool claim are
        # durable, but the native first-start CAS has not been attempted.
        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)

    name = "web-recovered-reserved-child"
    features = RuntimeFeatures(name)
    features.model("parent", _model("parent", parent))
    child_model = _model("child", child)
    child_model.contract["input_mime_types"] = ["text/plain"]
    features.model("child", child_model)
    features.capability(ExecutionCapability(retain_history=True))
    features.capability(StepPersistenceCapability())
    features.capability(FileSystemCapability(workspace))
    features.agent("default", model="parent", allow_subagents=("child",), allow_tools=(), allow_skills=())
    features.agent("child", model="child", allow_subagents=(), allow_tools=(), allow_skills=())
    storage = PersistenceStorage.from_root(tmp_path / "runtime")

    @asynccontextmanager
    async def open_host():
        async with (
            Persistence.open(name, storage=storage) as state,
            Runtime.open(persistence=state, features=features,
                         authorization=_HostAuthorization(), auto_recover=False) as runtime,
        ):
            app = create_app(runtime=runtime, local_workspace=True)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765") as http:
                yield state, runtime, http

    params = {"session_id": _SESSION}
    payload = {"prompt": "Delegate once", "request_id": "original-turn"}
    async def stop_before_reservation(_proof, _repository, _request, *, authority):
        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)

    missing = outcome == "reservation-missing"
    with monkeypatch.context() as fault:
        if missing:
            fault.setattr(NativeSubagentAdmission, "reserve", stop_before_reservation)
        else:
            fault.setattr(NativeSubagentAdmission, "commit_start", stop_before_start)
        async with open_host() as (state, runtime, http):
            create = await http.post("/api/sessions", headers=_HEADERS,
                                     json={"session_id": _SESSION, "request_id": "create"})
            assert create.status_code == 201, create.text
            posted = await http.post("/api/session/messages", params=params, headers=_HEADERS, json=payload)
            assert posted.status_code == 202, posted.text
            turn_id = posted.json()["turn_id"]
            key = SessionTurnKey(_SESSION, turn_id)

            async def original_tool():
                values = await state.checkpoint.tools.list_by_execution(turn_id, tenant_id=runtime.tenant_id)
                return next((value for value in values if value.tool_call_id == invocation), None)

            original = await _wait_for_committed(
                original_tool, lambda value: value is not None and value.status is ToolOperationStatus.EFFECT_UNKNOWN,
            )
            children = await state.execution.executions.list_children(turn_id, tenant_id=runtime.tenant_id)
            if missing:
                assert not children
                reserved = None
            else:
                [reserved] = children
                assert reserved.status is ExecutionStatus.PENDING_START and reserved.started_at is None
            assert calls == ["parent dispatch"]
            await runtime.handoff()

    attachment.unlink()
    child_rule.write_text("CHANGED-CHILD-RULE", encoding="utf-8")
    request_key = canonical_sha256({"kind": "subagent", "parent_execution_id": turn_id, "invocation_id": invocation})
    async with open_host() as (state, runtime, http):
        turns = state.session_turns
        tenant_id = runtime.tenant_id
        initial = await state.execution.idempotency.get("execution.subagent", request_key, tenant_id=tenant_id)
        if missing:
            assert initial is None
        else:
            assert (initial.resource_id, initial.status) == (reserved.execution_id, IdempotencyStatus.RESERVED)
        async with turns._context.engine.begin() as connection:
            changed = await connection.execute(
                turns._turns.update().filter_by(**turns._turn_values(key), generation=1, state="accepted")
                .values(lease_until=0),
            )
            assert changed.rowcount == 1

        def no_new_reservation(*_args, **_kwargs):
            pytest.fail("recovery must not reserve another child")

        monkeypatch.setattr(state.execution.executions, "reserve_native_subagent", no_new_reservation)
        unknown_start = outcome == "unknown-start-no-dispatch"
        if unknown_start:
            commands = runtime._execution_service.runtime_backend()._runtime_commands
            commit = commands.commit_recovered_native_subagent_start

            async def lose_start_ack(*args, **kwargs):
                started = await commit(*args, **kwargs)
                assert started.execution_id == reserved.execution_id
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

            monkeypatch.setattr(commands, "commit_recovered_native_subagent_start", lose_start_ack)

        recovered = await http.post("/api/session/recover", params=params, headers=_HEADERS,
                                    json={"turn_id": turn_id, "request_id": "recover"})
        children = await state.execution.executions.list_children(turn_id, tenant_id=tenant_id)
        receipt = await state.execution.idempotency.get("execution.subagent", request_key, tenant_id=tenant_id)
        row = await turns.get_turn(key)
        settled = await state.checkpoint.tools.get_operation(original.tool_operation_id, tenant_id=tenant_id)
        if missing:
            assert recovered.status_code == 503 and recovered.json()["code"] == "STORAGE_RECOVERY_REQUIRED", recovered.text
            assert recovered.json()["safe_details"] == {"reason": "native_child_reservation_missing", "generation": 2}
            assert row["state"] == "accepted" and row["generation"] == 2 and settled == original
            assert not children and receipt is None
            assert calls == ["parent dispatch"] and returned_children == []
            return
        [same_child] = children
        assert receipt.resource_id == same_child.execution_id == reserved.execution_id
        assert receipt.request_digest == initial.request_digest
        assert row["generation"] == 2
        assert await state.execution.executions.get(turn_id, tenant_id=tenant_id) is None

        if unknown_start:
            assert recovered.status_code == 503 and recovered.json()["code"] == "STORAGE_RECOVERY_REQUIRED", recovered.text
            assert recovered.json()["safe_details"] == {"reason": "native_child_recovery_unconfirmed"}
            assert row["state"] == "accepted" and settled == original
            assert same_child.status is ExecutionStatus.STARTED and same_child.agent_run_seq == 0
            assert receipt.status is IdempotencyStatus.STARTED
            assert calls == ["parent dispatch"] and returned_children == []
        else:
            assert recovered.status_code == 200 and recovered.json()["accepted"], recovered.text
            assert row["state"] == "succeeded" and settled.status is ToolOperationStatus.COMPLETED
            assert same_child.status is ExecutionStatus.SUCCEEDED
            assert receipt.status is IdempotencyStatus.COMPLETED
            assert calls == ["parent dispatch", "child dispatch", "parent continuation"]
            assert returned_children == [reserved.execution_id]
            replay = await http.post("/api/session/messages", params=params, headers=_HEADERS, json=payload)
            assert replay.json()["turn_id"] == turn_id
            result = await runtime.history.turn_result(_SESSION, turn_id, principal=runtime.default_principal)
            assert result["output"] == {"text": "completed from retained delegation"}
            assert calls == ["parent dispatch", "child dispatch", "parent continuation"]
