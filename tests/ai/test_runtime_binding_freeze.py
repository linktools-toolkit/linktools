#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for execution-owned binding dependency snapshots."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from linktools.ai.agent import (
    AgentBinding,
    AgentBindingSnapshot,
    AgentCatalog,
    AgentCompiler,
)
from linktools.ai.capability import (
    CapabilityContribution,
    FrozenSkillResourceSource,
    LocalSkillResourceSource,
    SkillDefinition,
    SkillSourceRef,
    SkillSourceRegistry,
)
from linktools.ai.core import Principal
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime._binding_freeze import _RuntimeBindingFreezer
from linktools.ai.runtime._context import RuntimeContext
from linktools.ai.runtime._runtime_service import Runtime
from linktools.ai.runtime._task_capability_snapshot import TaskCapabilitySnapshotStore
from linktools.ai.runtime.service_api import ExecutionHandle, ExecutionRequest
from linktools.ai.spec import AgentSpec, SkillSpec
from linktools.ai.storage import InMemoryObjectStore
from linktools.ai.task import (
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphLimits,
    TaskGraphRequest,
    TaskNode,
)


@dataclass(frozen=True)
class _BindingFixture:
    compiler: AgentCompiler
    catalog: AgentCatalog
    freezer: _RuntimeBindingFreezer
    objects: InMemoryObjectStore
    binding: AgentBinding
    resource: Path


class _RecordingExecution:
    def __init__(self) -> None:
        self.binding_digest: str | None = None
        self.binding_snapshot: AgentBindingSnapshot | None = None

    async def start(
        self,
        binding_digest: str,
        request: ExecutionRequest,
        *,
        dependency_hold_id: str | None = None,
        binding_snapshot: AgentBindingSnapshot | None = None,
    ) -> ExecutionHandle:
        del request, dependency_hold_id
        self.binding_digest = binding_digest
        self.binding_snapshot = binding_snapshot
        return ExecutionHandle("execution")


def _fixture(tmp_path: Path) -> _BindingFixture:
    skill_root = tmp_path / "skills"
    package = skill_root / "child-skill"
    package.mkdir(parents=True)
    resource = package / "guide.txt"
    resource.write_text("original", encoding="utf-8")

    candidates = (
        CapabilityContribution.from_declaration(
            SkillDefinition(
                SkillSpec("child-skill", "Use the child guide."),
                SkillSourceRef("application", "child-skill"),
            )
        ),
        CapabilityContribution.from_declaration(
            SkillDefinition(
                SkillSpec("unreachable-skill", "Not reachable from a child execution."),
                SkillSourceRef("missing", "unreachable-skill"),
            )
        ),
    )
    specs = {
        "parent": AgentSpec(
            "parent",
            allow_skills=(),
            allow_subagents=("child",),
        ),
        "child": AgentSpec(
            "child",
            allow_skills=("child-skill",),
            allow_subagents=("grandchild",),
        ),
        "grandchild": AgentSpec(
            "grandchild",
            allow_skills=("unreachable-skill",),
            allow_subagents=(),
        ),
    }
    compiler = AgentCompiler(
        model_resolver=ModelRegistry.openai(model="gpt-test").snapshot(),
        candidates=candidates,
        agents=specs,
    )
    catalog = AgentCatalog(
        {
            agent_id: compiler.compile(spec)
            for agent_id, spec in specs.items()
        }
    )
    objects = InMemoryObjectStore("execution")
    freezer = _RuntimeBindingFreezer(
        catalog,
        compiler,
        SkillSourceRegistry(
            (LocalSkillResourceSource("application", skill_root),)
        ),
        objects,
    )
    return _BindingFixture(
        compiler,
        catalog,
        freezer,
        objects,
        compiler.bind(catalog.root_definition("parent")),
        resource,
    )


def _frozen_child(snapshot: AgentBindingSnapshot) -> AgentBindingSnapshot:
    assert snapshot.subagent_ids == ("child",)
    assert len(snapshot.subagent_bindings) == 1
    child = snapshot.subagent_bindings[0]
    assert child.agent_spec.id == "child"
    assert child.subagents == ()
    assert child.subagent_bindings == ()
    return child


def _skill_snapshot(child: AgentBindingSnapshot):
    pin = next(item for item in child.selected if item.kind == "skill")
    skill = SkillDefinition.from_semantic_contract(pin.contract)
    assert skill.source_ref is not None
    assert skill.source_ref.snapshot is not None
    return skill.source_ref.snapshot


@pytest.mark.asyncio
async def test_binding_freeze_captures_only_direct_child_resources(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    frozen = await fixture.freezer.freeze(fixture.binding)
    snapshot = _skill_snapshot(_frozen_child(frozen.snapshot))

    assert snapshot.store_id == fixture.objects.store_id
    assert await fixture.objects.stat(snapshot.key) is not None

    fixture.resource.write_text("changed", encoding="utf-8")
    source = FrozenSkillResourceSource(
        "application",
        {"child-skill": snapshot},
        fixture.objects,
    )
    assert await source.read("child-skill", "guide.txt") == b"original"


@pytest.mark.asyncio
async def test_task_capture_does_not_build_static_root_closure(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    graph = TaskGraph(
        "graph",
        (
            TaskNode(
                "root",
                input={
                    "type": "linktools.ai.agent",
                    "version": 1,
                    "binding": fixture.binding.snapshot.to_payload(),
                },
            ),
        ),
    )
    admission = TaskGraphAdmission.from_request(
        TaskGraphRequest(
            graph,
            Principal("principal", "tenant"),
            "task-capture",
            TaskGraphLimits(),
        )
    )
    snapshots = TaskCapabilitySnapshotStore(
        "namespace",
        fixture.compiler,
        fixture.freezer,
        InMemoryObjectStore("task"),
        agent_task_type="linktools.ai.agent",
    )

    frozen = await snapshots.capture(admission, graph)

    assert frozen.roots == {}
    snapshot = _skill_snapshot(
        _frozen_child(frozen.bindings[fixture.binding.digest])
    )
    assert snapshot.store_id == fixture.objects.store_id


@pytest.mark.asyncio
async def test_runtime_start_admits_frozen_binding(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    execution = _RecordingExecution()
    runtime = Runtime(
        fixture.catalog,
        fixture.compiler,
        execution,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        None,
        namespace="namespace",
        context=RuntimeContext(None),
        binding_freezer=fixture.freezer,
    )

    started = await runtime._start_for_agent(
        fixture.catalog.root_definition("parent").digest,
        "prompt",
        files=(),
        output=None,
        principal=None,
        session_id=None,
        idempotency_key="runtime-freeze",
        memory_scope=None,
        mode="run",
        planning=None,
        thinking=None,
    )

    assert started.execution_id == "execution"
    assert execution.binding_snapshot is not None
    assert execution.binding_digest == execution.binding_snapshot.binding_digest
    snapshot = _skill_snapshot(_frozen_child(execution.binding_snapshot))
    assert snapshot.store_id == fixture.objects.store_id
