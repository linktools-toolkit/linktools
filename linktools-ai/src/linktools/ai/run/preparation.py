#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RunPreparationCoordinator: the single owner of RunDefinitionSnapshot creation.

Every run entry point (Runtime agent run / run_stream / swarm run) builds the
immutable run-definition snapshot through this coordinator so the snapshot's
shape, fingerprint, and identity restoration are defined once -- not duplicated
across Runtime and SwarmRunner (the prior double-create). A run that cannot be
persisted for resume is rejected here rather than silently resuming from a
caller-supplied spec later."""

import hashlib
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Mapping

from ..json import canonical_json
from ..utils.freeze import freeze_value
from .definition import (
    RunDefinitionSnapshot,
    serialize_agent_spec,
    serialize_swarm_spec,
    spec_fingerprint,
)

if TYPE_CHECKING:
    from .context import RunContext
    from .definition import RunDefinitionStore


class RunPreparationCoordinator:
    """Owns RunDefinitionSnapshot creation for every run entry point."""

    def __init__(self, run_definitions: "RunDefinitionStore") -> None:
        self._run_definitions = run_definitions

    async def prepare_agent_run(
        self, *, spec: Any, context: "RunContext"
    ) -> RunDefinitionSnapshot:
        """Build + persist the snapshot for an agent run. The fingerprint is the
        canonical-JSON hash of the serialized spec, so resume detects tampering
        or a drifted output_schema. serialized_spec + manifest are deep-frozen
        so the caller mutating its own dict after prepare cannot affect the
        persisted snapshot."""
        snapshot = RunDefinitionSnapshot(
            run_id=context.run_id,
            runnable_type=str(context.runnable_type.value),
            runnable_id=context.runnable_id,
            serialized_spec=freeze_value(serialize_agent_spec(spec)),
            spec_fingerprint=spec_fingerprint(spec),
            user_id=context.user_id,
            tenant_id=context.tenant_id,
            workspace=context.workspace,
            provider_revision=None,
            created_at=datetime.now(timezone.utc),
            manifest=_manifest(spec),
        )
        await self._run_definitions.create(snapshot)
        return snapshot

    async def prepare_swarm_run(
        self,
        *,
        spec: Any,
        members: "Mapping[str, Any]",
        context: "RunContext",
    ) -> RunDefinitionSnapshot:
        """Build + persist the snapshot for a swarm run. The serialized form
        carries the canonical swarm spec plus each member agent spec, and the
        fingerprint covers both so a changed member invalidates resume."""
        swarm_serialized = {
            "type": "swarm",
            "spec": serialize_swarm_spec(spec),
            "members": {aid: serialize_agent_spec(a) for aid, a in members.items()},
        }
        swarm_fp = hashlib.sha256(canonical_json(swarm_serialized).encode()).hexdigest()
        snapshot = RunDefinitionSnapshot(
            run_id=context.run_id,
            runnable_type=str(context.runnable_type.value),
            runnable_id=context.runnable_id,
            serialized_spec=freeze_value(swarm_serialized),
            spec_fingerprint=swarm_fp,
            user_id=context.user_id,
            tenant_id=context.tenant_id,
            workspace=context.workspace,
            provider_revision=None,
            created_at=datetime.now(timezone.utc),
            manifest=_manifest(spec),
        )
        await self._run_definitions.create(snapshot)
        return snapshot


def _manifest(spec: Any) -> "Mapping[str, Any]":
    """The execution manifest: revisions of the bundle/policy/capabilities the
    run was prepared against. Captures what is cheaply available now (the spec
    id + a frozen snapshot of the tool declarations); richer provider/MCP/skill
    revisions are layered in as callers expose them. Frozen so it cannot drift
    after the snapshot is taken."""
    return freeze_value(
        {
            "runnable_id": getattr(spec, "id", None),
            "tools": [
                {"kind": getattr(t, "kind", None), "name": getattr(t, "name", None)}
                for t in (getattr(spec, "tools", None) or ())
            ],
        }
    )
