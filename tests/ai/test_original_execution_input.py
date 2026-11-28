#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tools and validators read original input without replacing live session state."""

import asyncio
import hashlib
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import ApprovalRequired
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.tools import RunContext

from linktools.ai.asset import AssetKey, AssetVersionRef
from linktools.ai.capability import AgentContext, ExecutionCapability, ReferencedInputsCapability, SessionCapability
from linktools.ai.core import ApprovalDecision, ExecutionStatus, Principal, TenantAuthorizationPolicy, ToolOperationStatus
from linktools.ai.execution import ApprovalDecisionRequest, ExecutionInputContext
from linktools.ai.inputs import ExecutionContextArchive
from linktools.ai.model import ReferencedContent
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.spec import CapabilityRef
from linktools.ai.storage import StorageEntryRevision

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


_BODY = b"retained source"
_SOURCE = CapabilityRef("capability", "original-file", 1)


def _reference() -> ReferencedContent:
    return ReferencedContent(AssetVersionRef(
        AssetKey("document", "original"), "test", StorageEntryRevision(1),
        hashlib.sha256(_BODY).hexdigest(), len(_BODY),
    ), media_type="text/plain", source=_SOURCE, representation="text",
        identifier="initial-file", source_parameters={"selected": ["original"]})


def _initial_reference(context: AgentContext[None]) -> dict[str, object]:
    return context.original_input["prompt"]["items"][1]["value"]["items"][0]


@pytest.mark.asyncio
@pytest.mark.parametrize("imported", (False, True))
async def test_original_input_is_detached_and_shared_by_tools_and_output_hooks(imported: bool) -> None:
    observed = []
    metadata = {"selected": ["original"]}
    imported_context = ExecutionInputContext.from_messages((), session_metadata=metadata)

    def inspect_input(context: AgentContext[None]) -> str:
        assert context.input_context.session_metadata == metadata
        assert context.input_context.model_messages() == ()
        descriptor = _initial_reference(context)
        assert descriptor["ref"] == _reference().ref.to_payload()
        assert descriptor["source"] == {"kind": "capability", "id": "original-file", "revision": 1}
        descriptor["source_parameters"]["selected"].append("changed")
        assert _initial_reference(context)["source_parameters"] == {"selected": ["original"]}
        with pytest.raises(TypeError):
            context.original_input["prompt"] = "changed"
        with pytest.raises(FrozenInstanceError):
            context.input_context = None
        observed.append(context)
        return "inspected"

    class ValidateInput(AbstractCapability):
        async def after_output_validate(
            self, ctx: RunContext[AgentContext[None]], *, output_context: Any, output: Any,
        ) -> Any:
            assert observed and ctx.deps.original_input == observed[0].original_input
            assert ctx.deps.input_context == observed[0].input_context
            return output

    async def load(context: AgentContext[None], reference: ReferencedContent) -> bytes:
        assert _initial_reference(context)["ref"] == reference.ref.to_payload()
        return _BODY

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[ToolCallPart("inspect_input", {}, tool_call_id="inspect")])
        return ModelResponse(parts=[TextPart("done")])

    features = RuntimeFeatures("original-input")
    features.tool(inspect_input)
    features.capability(ValidateInput(), id="input-check", revision=1, scope="agent")
    features.capability(ReferencedInputsCapability(load), id=_SOURCE.id, revision=1, scope="agent")
    features.agent("default", allow_tools=("inspect_input",), allow_skills=(), allow_subagents=())
    async with Persistence.open("original-input", storage=PersistenceStorage.in_memory()) as persistence:
        async with Runtime.open(persistence=persistence, features=model_features(
            _Models(_UsageFunctionModel(model)), features,
        )) as runtime:
            if imported:
                execution = await runtime.agents.get().start(["inspect", _reference()], input_context=imported_context)
            else:
                session = await runtime.agents.get().create_session("input-session", metadata=metadata)
                execution = await session.start(["inspect", _reference()])
            result = (await execution.wait(timeout_seconds=15)).result
            assert result.status is ExecutionStatus.SUCCEEDED, result
            assert result.output == {"text": "done"}
    assert len(observed) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("local_sqlite", "sqlite"))
async def test_original_input_reopens_while_session_metadata_remains_live(tmp_path: Path, backend: str) -> None:
    observed = []
    approval_requested = False
    principal = Principal("original-actor", "default", "service")
    recovery_actor = Principal("recovery-actor", "default", "service")

    class HostAuthorization(TenantAuthorizationPolicy):
        """The embedded host authorizes both the initiator and the approver."""

    def inspect_input(context: AgentContext[None]) -> str:
        nonlocal approval_requested
        if not approval_requested:
            approval_requested = True
            raise ApprovalRequired()
        observed.append(context)
        return "inspected"

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[ToolCallPart("inspect_input", {}, tool_call_id="inspect")])
        return ModelResponse(parts=[TextPart("done")])

    async def load(context: AgentContext[None], reference: ReferencedContent) -> bytes:
        assert context.principal == principal
        assert _initial_reference(context)["ref"] == reference.ref.to_payload()
        return _BODY

    def features() -> RuntimeFeatures:
        value = RuntimeFeatures("original-input-reopen")
        value.tool(inspect_input, effect_policy="replay_safe")
        value.capability(ExecutionCapability(retain_history=True))
        value.capability(SessionCapability(checkpoints=True))
        value.capability(ReferencedInputsCapability(load), id=_SOURCE.id, revision=1, scope="agent")
        value.agent("default", allow_tools=("inspect_input",), allow_skills=(), allow_subagents=())
        return model_features(_Models(_UsageFunctionModel(model)), value)

    async def start(runtime):
        return await runtime.sessions.start(
            "session", ["original request", _reference()], agent="default", principal=principal,
            idempotency_key="inspect-original",
        )

    storage = (PersistenceStorage.from_root(tmp_path / "state", exclusive_writer=True)
               if backend == "local_sqlite" else PersistenceStorage.sqlite(tmp_path / "state.db"))
    async with Persistence.open("original-input-reopen", storage=storage) as persistence:
        async with Runtime.open(persistence=persistence, features=features(),
                                authorization=HostAuthorization()) as runtime:
            await runtime.agents.get().create_session(
                "session", principal=principal, metadata={"selected": ["original"]},
            )
            turn = await start(runtime)
            key = turn.key
            assert (await asyncio.wait_for(turn.wait(), 15)).status is ExecutionStatus.WAITING_DEFERRED
            [pending] = await runtime.approvals.list(key, principal=principal)
            [original] = await persistence.checkpoint.tools.list_by_execution(key.turn_id, tenant_id=runtime.tenant_id)
            assert original.status is ToolOperationStatus.PENDING
            await runtime.handoff()

    async with Persistence.open("original-input-reopen", storage=storage) as persistence:
        async with Runtime.open(persistence=persistence, features=features(),
                                authorization=HostAuthorization()) as runtime:
            session = await runtime.sessions.get("session", principal=principal)
            current = await session.inspect()
            assert current["active_turn_id"] == key.turn_id
            assert not await session.close(expected_revision=current["revision"])
            assert not await session.update(
                expected_revision=current["revision"], metadata={"selected": ["ignored"]}, replace_cwd=True,
            )
            assert await session.update(expected_revision=current["revision"], metadata={"selected": ["current"]})
            assert not await session.update(expected_revision=current["revision"], metadata={"selected": ["stale"]})
            replay = await start(runtime)
            assert replay.key == key
            await runtime.approvals.decide(key, ApprovalDecisionRequest(
                recovery_actor, pending.approval_id, "approve", ApprovalDecision.APPROVE,
            ))
            result = await replay.wait_final(timeout_seconds=15)
            assert result.status is ExecutionStatus.SUCCEEDED, result
            [completed] = await persistence.checkpoint.tools.list_by_execution(key.turn_id, tenant_id=runtime.tenant_id)
            assert completed.tool_operation_id == original.tool_operation_id
            assert completed.status is ToolOperationStatus.COMPLETED
    assert len(observed) == 1
    context = observed[0]
    assert context.principal == principal and context.tool_call.principal == principal
    assert context.session_metadata == {"selected": ["current"]}
    assert context.input_context.session_metadata == {"selected": ["original"]}
    assert context.original_input["prompt"]["items"][0] == {"kind": "text", "text": "original request"}
    assert _initial_reference(context)["ref"] == _reference().ref.to_payload()
    assert _initial_reference(context)["source_parameters"] == {"selected": ["original"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "local_sqlite", "sqlite"))
async def test_preload_skill_descriptors_round_trip_in_the_original_context_owner(tmp_path: Path, backend: str) -> None:
    descriptor = {"skill": {"kind": "skill", "id": "preload", "revision": 1},
                  "declaration": {"ref": _reference().ref.to_payload(), "path": "SKILL.md"}}
    context = ExecutionInputContext.from_messages((), preload_skills=(descriptor,))
    descriptor["skill"]["id"] = "changed"
    assert context.preload_skills[0]["skill"]["id"] == "preload"
    payload = context.to_payload()
    assert payload["preload_skills"] == [dict(context.preload_skills[0])]
    assert context.digest != ExecutionInputContext.from_messages(()).digest
    assert ExecutionInputContext.from_payload(payload) == context
    storage = (PersistenceStorage.in_memory() if backend == "memory" else
               PersistenceStorage.from_root(tmp_path / "state", exclusive_writer=True) if backend == "local_sqlite" else
               PersistenceStorage.sqlite(tmp_path / "state.db"))
    async with Persistence.open("preload-input", storage=storage) as persistence:
        archive = ExecutionContextArchive(
            persistence.execution.executions.state_store,
            persistence.archive_object_store(PersistenceDomain.EXECUTION),
            namespace=persistence.namespace, tenant_id=persistence.tenant_id,
        )
        prepared = await archive.prepare(context)
        assert len(prepared.reference.payload.decode()["preload_skills"]) == 1
        await prepared.archive.store.mutate(lambda tx: prepared.archive.publish(tx, prepared.records))
        loaded = await archive.load(prepared.reference)
        assert loaded == context and loaded.digest == context.digest
        loaded.preload_skills[0]["skill"]["id"] = "detached"
        assert loaded.preload_skills[0]["skill"]["id"] == "preload"
