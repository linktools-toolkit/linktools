#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authorized execution metadata exposes bound identities without loading content."""

from collections import Counter
from dataclasses import FrozenInstanceError, asdict, replace
from datetime import datetime, timezone

import pytest

from linktools.ai.agent import AgentBindingContract
from linktools.ai.capability import CapabilityContribution, RuntimeFeatures, SkillDefinition
from linktools.ai.core import AuthorizationAction, ExecutionLineageKind, ExecutionStatus, Principal, ResourceKind, ResourceRef
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import AgentTaskInput, ExecutionInfo, Runtime, RuntimeHistory, RuntimeStorage, UsageSummary
from linktools.ai.runtime.state._contracts import ExecutionCandidate, ExecutionCandidatePage
from linktools.ai.spec import AgentSpec, CapabilityRef, SkillSpec, capability_ref_payload
from linktools.ai.task import Task, TaskGraph, TaskNode

from ._runtime_test_helpers import RuntimeUsageModels
from .test_capability_filter import _compiler, contributions
from .test_runtime_binding_contract import _binding_contract, _execution
from .test_runtime_history_public import _ResultExecutions


class _Repository:
    def __init__(self, record, calls):
        self.record = record
        self.calls = calls

    async def get_header(self, execution_id, *, tenant_id):
        self.calls.append("header")
        if tenant_id != "tenant":
            return None
        return ResourceRef(ResourceKind.EXECUTION, execution_id, tenant_id)

    async def get(self, execution_id, *, tenant_id):
        self.calls.append("record")
        return self.record

    async def list_candidates(self, *, tenant_id, **kwargs):
        self.calls.append("candidates")
        items = () if tenant_id != "tenant" else (ExecutionCandidate(self.record, "cursor"),)
        return ExecutionCandidatePage(items, False)


class _Authorization:
    def __init__(self, calls, allowed=True):
        self.calls = calls
        self.allowed = allowed

    async def authorize(self, principal, action, resource):
        self.calls.append("authorize")
        assert action is AuthorizationAction.EXECUTION_READ
        if not self.allowed:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)


class _Service:
    def __init__(self, calls):
        self.calls = calls

    async def usage(self, execution_id, *, principal):
        self.calls.append("usage")
        return UsageSummary()


def _history(record, *, allowed=True):
    calls = []
    return RuntimeHistory(
        _Service(calls), tenant_id="tenant", executions=_Repository(record, calls),
        authorization=_Authorization(calls, allowed),
    ), calls


@pytest.mark.asyncio
async def test_safe_bound_refs_cover_every_kind_without_extra_reads(contributions):
    binding = _compiler(contributions).bind(
        _compiler(contributions).compile(AgentSpec("root", instructions=("private prompt",)))
    ).binding_contract
    history, calls = _history(_execution(binding=binding))
    principal = Principal("caller", "tenant", "service")
    info = await history.inspect_execution("execution", principal=principal)
    expected = tuple((pin.kind, pin.id, pin.revision) for pin in binding.selected) + (("agent", "child", 3),)
    assert info.capability_refs == binding.capability_refs
    assert tuple((ref.kind, ref.id, ref.revision) for ref in info.capability_refs) == expected
    assert {ref.kind for ref in info.capability_refs} == {"agent", "tool", "skill", "mcp", "capability"}
    assert info.agent_id == "root"
    assert all(ref.id != "root" for ref in info.capability_refs)
    assert all(set(asdict(ref)) == {"kind", "id", "revision"} for ref in info.capability_refs)
    assert "private prompt" not in str(asdict(info))
    assert "Delegate work" not in str(asdict(info))
    assert Counter(calls) == Counter(header=1, authorize=1, record=1, usage=1)
    assert calls.index("authorize") < calls.index("record")
    with pytest.raises(FrozenInstanceError):
        info.capability_refs[0].id = "changed"

    calls.clear()
    recent = await history.recent_executions(principal=principal)
    assert recent[0].capability_refs == info.capability_refs
    assert Counter(calls) == Counter(candidates=1, authorize=1)
    old_payload = binding.to_payload()
    old_payload.pop("subagent_bindings", None)
    old_history, _ = _history(_execution(binding=AgentBindingContract.from_payload(old_payload)))
    assert (await old_history.inspect_execution("execution", principal=principal)).capability_refs == info.capability_refs


@pytest.mark.asyncio
async def test_bound_refs_share_public_identity_and_restore_without_current_filter(contributions):
    compiler = _compiler(contributions)
    bound = compiler.bind(compiler.compile(AgentSpec("root")))
    contract = bound.binding_contract
    payload = contract.to_payload()
    digest = contract.binding_digest
    refs = contract.capability_refs
    assert all(type(ref) is CapabilityRef for ref in refs)
    assert tuple(asdict(ref) for ref in refs if ref.kind != "agent") == tuple(
        capability_ref_payload(pin.kind, pin.id, pin.contract) for pin in contract.selected
    )

    def unexpected(agent, item):
        raise AssertionError("restored identities must not use the current selection filter")

    decoded = AgentBindingContract.from_payload(payload)
    restored = _compiler(contributions, unexpected).restore(decoded)
    assert restored.binding_contract.capability_refs == refs
    assert decoded.to_payload() == payload and decoded.binding_digest == digest
    assert contract.to_payload() == payload and contract.binding_digest == digest
    history, _ = _history(_execution(binding=restored.binding_contract))
    principal = Principal("caller", "tenant", "service")
    assert (await history.inspect_execution("execution", principal=principal)).capability_refs == refs


@pytest.mark.asyncio
async def test_bound_refs_preserve_generic_capability_order_without_recursing_children(contributions):
    compiler = _compiler(contributions)
    bound = compiler.bind(compiler.compile(AgentSpec("root"))).binding_contract
    extra = next(pin for pin in bound.selected if pin.kind == "capability")
    other = replace(extra, id="aaa-other")
    child = compiler.bind_subagent(compiler.compile(AgentSpec("child", revision=3))).binding_contract
    contract = replace(bound, selected=(*bound.selected, other), subagent_bindings=(child,))
    assert [ref.id for ref in contract.capability_refs if ref.kind == "capability"] == ["extra", "aaa-other"]
    assert contract.capability_refs[-1] == CapabilityRef("agent", "child", 3)
    assert len(contract.capability_refs) == len(contract.selected) + len(contract.subagents)
    assert contract.capability_refs == replace(contract, subagent_bindings=()).capability_refs


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ("inspect", "recent"))
@pytest.mark.parametrize("tenant,allowed", (("tenant", False), ("other", True)))
async def test_capability_refs_remain_behind_execution_read_authorization(entry, tenant, allowed):
    history, calls = _history(_execution(), allowed=allowed)
    principal = Principal("caller", tenant, "service")
    if entry == "inspect":
        with pytest.raises(AIError) as error:
            await history.inspect_execution("execution", principal=principal)
        assert error.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert "record" not in calls and "usage" not in calls
    else:
        assert await history.recent_executions(principal=principal) == ()
        assert "record" not in calls and "usage" not in calls


@pytest.mark.asyncio
async def test_empty_old_agent_binding_and_task_metadata_remain_distinct():
    # Earlier Agent bindings omit captured child contracts; their pins remain authoritative.
    payload = _binding_contract().to_payload()
    payload.pop("subagent_bindings", None)
    restored = AgentBindingContract.from_payload(payload)
    agent, _ = _history(_execution(binding=restored))
    task, _ = _history(_ResultExecutions().record)
    principal = Principal("caller", "tenant", "service")
    assert (await agent.inspect_execution("execution", principal=principal)).capability_refs == ()
    assert (await task.inspect_execution("execution", principal=principal)).capability_refs is None


def test_execution_info_earlier_constructor_keeps_optional_refs_default():
    now = datetime.now(timezone.utc)
    info = ExecutionInfo("execution", "agent", "root", None, ExecutionStatus.SUCCEEDED,
                         ExecutionLineageKind.RUN, None, "execution", None, None, now, now, None)
    assert info.capability_refs is None
    assert asdict(CapabilityRef("skill", "guide", 2)) == {"kind": "skill", "id": "guide", "revision": 2}


@pytest.mark.asyncio
async def test_graph_agent_refs_preserve_derived_binding_after_catalog_change_and_reopen(tmp_path):
    group = RuntimeFeatures("bound-ref-test")
    group.agent("root", allow_tools=("business",), allow_skills=("guide",), allow_subagents=())

    def business(context: object) -> str:
        return "unused"

    group.tool(business, revision=3)
    capture = await group.capture()
    capture = replace(capture, contributions=(*capture.contributions,
        CapabilityContribution.from_declaration(SkillDefinition(SkillSpec("guide", "Private guide", revision=2)))))
    path = tmp_path / "runtime.db"

    async def prepare(context):
        return {"prepared": True}

    application_task = Task("bound-ref-prepare", prepare, effect_policy="none")
    async with Runtime.open("bound-refs", models=RuntimeUsageModels(),
                            storage=RuntimeStorage.sqlite(path), features=capture) as runtime:
        derived = runtime.agents.get("root").derive(allow_tools=())
        task = runtime.tasks.from_agent("bound-ref-agent", derived)
        run = await runtime.tasks.bind(application_task, task).start(
            TaskGraph("bound-ref-graph", (TaskNode("prepare", (), task=application_task.ref),
                TaskNode("agent", ("prepare",), task=task, input=AgentTaskInput("hello")))),
            idempotency_key="bound-ref-graph",
        )
        await run.wait(timeout_seconds=10)
        state = await run.state()
        nodes = {node.node_id: node for node in state.node_states}
        execution_id = nodes["agent"].execution_id
        assert execution_id is not None
        principal = runtime.default_principal
        task_info = await runtime.history.inspect_execution(nodes["prepare"].execution_id, principal=principal)
        assert task_info.binding_kind == "task" and task_info.capability_refs is None
        info = await runtime.history.inspect_execution(execution_id, principal=principal)
        assert info.capability_refs == (CapabilityRef("skill", "guide", 2),)
        assert info.agent_id == "root"

    changed = RuntimeFeatures("bound-ref-test")
    changed.agent("root", revision=2, allow_tools=(), allow_skills=(), allow_subagents=())
    async with Runtime.open("bound-refs", models=RuntimeUsageModels(),
                            storage=RuntimeStorage.sqlite(path), features=changed) as current:
        assert current.agents.get("root").derive(model="default").compiled.skill_definitions == ()
        assert (await current.history.inspect_execution(execution_id, principal=principal)).capability_refs == info.capability_refs
    async with RuntimeHistory.open("bound-refs", storage=RuntimeStorage.sqlite(path)) as history:
        reopened = await history.inspect_execution(execution_id, principal=principal)
        assert reopened.capability_refs == info.capability_refs
        assert (await history.recent_executions(principal=principal))[0].capability_refs == info.capability_refs
