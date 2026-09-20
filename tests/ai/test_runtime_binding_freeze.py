#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for execution-owned binding dependency snapshots."""

from pathlib import Path

import pytest

from linktools.ai.agent import AgentBindingSnapshot, AgentCatalog, AgentCompiler
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


def _compiler(
    specs: dict[str, AgentSpec],
    candidates: tuple[CapabilityContribution[object], ...],
) -> AgentCompiler:
    return AgentCompiler(
        model_resolver=ModelRegistry.openai(model="gpt-test").snapshot(),
        candidates=candidates,
        agents=specs,
    )


@pytest.mark.asyncio
async def test_binding_freeze_captures_direct_child_resources_only(
    tmp_path: Path,
) -> None:
    skill_root = tmp_path / "skills"
    package = skill_root / "child-skill"
    package.mkdir(parents=True)
    resource = package / "guide.txt"
    resource.write_text("original", encoding="utf-8")

    child_skill = SkillDefinition(
        SkillSpec("child-skill", "Use the child guide."),
        SkillSourceRef("application", "child-skill"),
    )
    unreachable_skill = SkillDefinition(
        SkillSpec("unreachable-skill", "Never loaded by the direct child."),
        SkillSourceRef("missing", "unreachable-skill"),
    )
    candidates = (
        CapabilityContribution.from_declaration(child_skill),
        CapabilityContribution.from_declaration(unreachable_skill),
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
    compiler = _compiler(specs, candidates)
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

    frozen = await freezer.freeze(
        compiler.bind(catalog.root_definition("parent"))
    )

    assert frozen.snapshot.subagent_ids == ("child",)
    assert len(frozen.snapshot.subagent_bindings) == 1
    child = frozen.snapshot.subagent_bindings[0]
    assert child.agent_spec.id == "child"
    assert child.subagents == ()
    assert child.subagent_bindings == ()

    child_pin = next(pin for pin in child.selected if pin.kind == "skill")
    frozen_skill = SkillDefinition.from_semantic_contract(child_pin.contract)
    assert frozen_skill.source_ref is not None
    assert frozen_skill.source_ref.snapshot is not None
    snapshot = frozen_skill.source_ref.snapshot
    assert snapshot.store_id == objects.store_id
    assert await objects.stat(snapshot.key) is not None

    resource.write_text("changed", encoding="utf-8")
    frozen_source = FrozenSkillResourceSource(
        "application",
        {"child-skill": snapshot},
        objects,
    )
    assert await frozen_source.read("child-skill", "guide.txt") == b"original"


@pytest.mark.asyncio
async def test_binding_freeze_is_idempotent_for_frozen_dependencies(
    tmp_path: Path,
) -> None:
    skill_root = tmp_path / "skills"
    package = skill_root / "review"
    package.mkdir(parents=True)
    (package / "notes.txt").write_text("v1", encoding="utf-8")

    skill = SkillDefinition(
        SkillSpec("review", "Review the notes."),
        SkillSourceRef("application", "review"),
    )
    specs = {
        "agent": AgentSpec(
            "agent",
            allow_skills=("review",),
            allow_subagents=(),
        )
    }
    compiler = _compiler(
        specs,
        (CapabilityContribution.from_declaration(skill),),
    )
    catalog = AgentCatalog({"agent": compiler.compile(specs["agent"])})
    objects = InMemoryObjectStore("execution")
    freezer = _RuntimeBindingFreezer(
        catalog,
        compiler,
        SkillSourceRegistry(
            (LocalSkillResourceSource("application", skill_root),)
        ),
        objects,
    )

    first = await freezer.freeze(
        compiler.bind(catalog.root_definition("agent"))
    )
    second = await freezer.freeze(first)

    assert second.snapshot == first.snapshot
    assert second.digest == first.digest


@pytest.mark.asyncio
async def test_task_capture_freezes_static_binding_without_root_closure(
    tmp_path: Path,
) -> None:
    skill_root = tmp_path / "skills"
    package = skill_root / "child-skill"
    package.mkdir(parents=True)
    (package / "guide.txt").write_text("child", encoding="utf-8")

    child_skill = SkillDefinition(
        SkillSpec("child-skill", "Use the child guide."),
        SkillSourceRef("application", "child-skill"),
    )
    unreachable_skill = SkillDefinition(
        SkillSpec("unreachable-skill", "Not reachable from a child execution."),
        SkillSourceRef("missing", "unreachable-skill"),
    )
    candidates = (
        CapabilityContribution.from_declaration(child_skill),
        CapabilityContribution.from_declaration(unreachable_skill),
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
    compiler = _compiler(specs, candidates)
    catalog = AgentCatalog(
        {
            agent_id: compiler.compile(spec)
            for agent_id, spec in specs.items()
        }
    )
    execution_objects = InMemoryObjectStore("execution")
    freezer = _RuntimeBindingFreezer(
        catalog,
        compiler,
        SkillSourceRegistry(
            (LocalSkillResourceSource("application", skill_root),)
        ),
        execution_objects,
    )
    binding = compiler.bind(catalog.root_definition("parent"))
    graph = TaskGraph(
        "graph",
        (
            TaskNode(
                "root",
                input={
                    "type": "linktools.ai.agent",
                    "version": 1,
                    "binding": binding.snapshot.to_payload(),
                },
            ),
        ),
    )
    request = TaskGraphRequest(
        graph,
        Principal("principal", "tenant"),
        "task-capture",
        TaskGraphLimits(),
    )
    admission = TaskGraphAdmission.from_request(request)
    snapshots = TaskCapabilitySnapshotStore(
        "namespace",
        compiler,
        freezer,
        InMemoryObjectStore("task"),
        agent_task_type="linktools.ai.agent",
    )

    frozen = await snapshots.capture(admission, graph)

    assert frozen.roots == {}
    captured = frozen.bindings[binding.digest]
    child = captured.subagent_bindings[0]
    assert child.agent_spec.id == "child"
    assert child.subagents == ()
    child_pin = next(pin for pin in child.selected if pin.kind == "skill")
    frozen_skill = SkillDefinition.from_semantic_contract(child_pin.contract)
    assert frozen_skill.source_ref is not None
    assert frozen_skill.source_ref.snapshot is not None
    assert frozen_skill.source_ref.snapshot.store_id == execution_objects.store_id


@pytest.mark.asyncio
async def test_runtime_start_admits_the_frozen_binding(tmp_path: Path) -> None:
    skill_root = tmp_path / "skills"
    package = skill_root / "review"
    package.mkdir(parents=True)
    (package / "notes.txt").write_text("runtime", encoding="utf-8")

    skill = SkillDefinition(
        SkillSpec("review", "Review the notes."),
        SkillSourceRef("application", "review"),
    )
    specs = {
        "agent": AgentSpec(
            "agent",
            allow_skills=("review",),
            allow_subagents=(),
        )
    }
    compiler = _compiler(
        specs,
        (CapabilityContribution.from_declaration(skill),),
    )
    definition = compiler.compile(specs["agent"])
    catalog = AgentCatalog({"agent": definition})
    objects = InMemoryObjectStore("execution")
    freezer = _RuntimeBindingFreezer(
        catalog,
        compiler,
        SkillSourceRegistry(
            (LocalSkillResourceSource("application", skill_root),)
        ),
        objects,
    )
    execution = _RecordingExecution()
    runtime = Runtime(
        catalog,
        compiler,
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
        binding_freezer=freezer,
    )

    started = await runtime._start_for_agent(
        definition.digest,
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
    pin = next(
        item
        for item in execution.binding_snapshot.selected
        if item.kind == "skill"
    )
    frozen_skill = SkillDefinition.from_semantic_contract(pin.contract)
    assert frozen_skill.source_ref is not None
    assert frozen_skill.source_ref.snapshot is not None
    assert frozen_skill.source_ref.snapshot.store_id == objects.store_id
