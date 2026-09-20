#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze execution-owned Agent binding dependencies."""

from collections.abc import Mapping
from dataclasses import replace
from typing import cast

from ..agent import (
    AgentBinding,
    AgentBindingSnapshot,
    AgentCatalog,
    AgentCompiler,
    SemanticPin,
)
from ..capability import (
    SkillDefinition,
    SkillSourceRegistry,
    SnapshotSkillResourceSource,
)
from ..errors import AIError, ErrorCode
from ..storage import ObjectRef, ObjectStore, StorageRevision


class _RuntimeBindingFreezer:
    """Materialize the immutable dependencies required by one execution binding."""

    def __init__(
        self,
        catalog: AgentCatalog,
        compiler: AgentCompiler,
        skill_sources: SkillSourceRegistry,
        object_store: ObjectStore,
        *,
        snapshot_resources: bool,
    ) -> None:
        if not isinstance(catalog, AgentCatalog):
            raise TypeError("catalog must be AgentCatalog")
        if not isinstance(compiler, AgentCompiler):
            raise TypeError("compiler must be AgentCompiler")
        if not isinstance(skill_sources, SkillSourceRegistry):
            raise TypeError("skill_sources must be SkillSourceRegistry")
        self._catalog = catalog
        self._compiler = compiler
        self._skill_sources = skill_sources
        self._objects = object_store
        self._snapshot_resources = snapshot_resources

    @property
    def root_ids(self) -> tuple[str, ...]:
        return self._catalog.root_ids

    async def freeze(self, binding: AgentBinding) -> AgentBinding:
        """Freeze one current binding before its first durable admission."""
        if not isinstance(binding, AgentBinding):
            raise TypeError("binding must be AgentBinding")
        snapshot = await self.freeze_snapshot(binding.snapshot)
        if snapshot == binding.snapshot:
            return binding
        return self._compiler.restore(snapshot)

    async def freeze_root(
        self,
        agent_id: str,
        *,
        skill_snapshots: "dict[tuple[str, str], ObjectRef] | None" = None,
    ) -> AgentBindingSnapshot:
        """Freeze one Agent as a root execution target and its direct children."""
        definition = self._catalog.root_definition(agent_id)
        return await self.freeze_snapshot(
            self._compiler.bind(definition, output=None).snapshot,
            skill_snapshots=skill_snapshots,
        )

    async def freeze_snapshot(
        self,
        snapshot: AgentBindingSnapshot,
        *,
        skill_snapshots: "dict[tuple[str, str], ObjectRef] | None" = None,
    ) -> AgentBindingSnapshot:
        """Freeze Skill resources and direct child bindings for one snapshot."""
        if not isinstance(snapshot, AgentBindingSnapshot):
            raise TypeError("snapshot must be AgentBindingSnapshot")
        cache = {} if skill_snapshots is None else skill_snapshots
        frozen = await self._freeze_skills(snapshot, skill_snapshots=cache)
        if frozen.subagent_bindings:
            children = tuple(
                await self._freeze_skills(child, skill_snapshots=cache)
                for child in frozen.subagent_bindings
            )
        else:
            children = tuple(
                await self._freeze_child(child_id, skill_snapshots=cache)
                for child_id in frozen.subagent_ids
            )
        return replace(frozen, subagent_bindings=children)

    async def _freeze_child(
        self,
        agent_id: str,
        *,
        skill_snapshots: "dict[tuple[str, str], ObjectRef]",
    ) -> AgentBindingSnapshot:
        definition = self._catalog.root_definition(agent_id)
        snapshot = self._compiler.bind_subagent(definition).snapshot
        return await self._freeze_skills(
            snapshot,
            skill_snapshots=skill_snapshots,
        )

    async def _freeze_skills(
        self,
        snapshot: AgentBindingSnapshot,
        *,
        skill_snapshots: "dict[tuple[str, str], ObjectRef]",
    ) -> AgentBindingSnapshot:
        selected: list[SemanticPin] = []
        for pin in snapshot.selected:
            if pin.kind != "skill":
                selected.append(pin)
                continue
            skill = SkillDefinition.from_semantic_contract(
                cast("Mapping[str, object]", pin.contract)
            )
            source_ref = skill.source_ref
            if (
                source_ref is None
                or source_ref.snapshot is not None
                or not self._snapshot_resources
            ):
                selected.append(pin)
                continue
            source = self._skill_sources.resolve(source_ref.source_id)
            if not isinstance(source, SnapshotSkillResourceSource):
                raise AIError(
                    ErrorCode.CAPABILITY_REQUIRED_MISSING,
                    safe_details={
                        "kind": "skill_snapshot",
                        "skill_id": skill.id,
                        "source_id": source_ref.source_id,
                    },
                )
            source_key = (source_ref.source_id, source_ref.root)
            reference = skill_snapshots.get(source_key)
            if reference is None:
                revision = await source.current_revision(source_ref.root)
                if not isinstance(revision, StorageRevision):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                reference = await source.snapshot(
                    source_ref.root,
                    expected_revision=revision,
                    object_store=self._objects,
                )
                skill_snapshots[source_key] = reference
            frozen_skill = SkillDefinition(
                skill.spec,
                source_ref.with_snapshot(reference),
            )
            selected.append(
                SemanticPin(
                    "skill",
                    pin.id,
                    frozen_skill.semantic_contract,
                )
            )
        return replace(snapshot, selected=tuple(selected))


__all__ = ["_RuntimeBindingFreezer"]
