#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RunPreparationCoordinator: the single owner of RunDefinitionSnapshot creation.

Every run entry point (Runtime agent run / run_stream / swarm run) builds the
immutable run-definition snapshot through this coordinator so the snapshot's
shape, fingerprint, and identity restoration are defined once -- not duplicated
across Runtime and SwarmRunner (the prior double-create). A run that cannot be
persisted for resume is rejected here rather than silently resuming from a
caller-supplied spec later.

The snapshot carries an :class:`ExecutionManifest` -- revisions / fingerprints
of the runnable, model, tools, and assets the run was prepared against. The
agent-run path compiles FIRST and threads the resolved model bundle in, so the
manifest's provider revision is populated; tool-handler revisions are still
None (handlers resolve at execution time, not prepare time). The resume path
runs the ManifestResolver against the persisted manifest and refuses a
NON_RESUMABLE snapshot or a drifted environment."""

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
from .manifest import Resumability, build_execution_manifest, manifest_to_dict

if TYPE_CHECKING:
    from .context import RunContext
    from .definition import RunDefinitionStore


class RunPreparationCoordinator:
    """Owns RunDefinitionSnapshot creation for every run entry point."""

    def __init__(self, run_definitions: "RunDefinitionStore") -> None:
        self._run_definitions = run_definitions

    async def prepare_agent_run(
        self,
        *,
        spec: Any,
        context: "RunContext",
        model_bundle: Any = None,
    ) -> RunDefinitionSnapshot:
        """Build + persist the snapshot for an agent run. The fingerprint is the
        canonical-JSON hash of the serialized spec, so resume detects tampering
        or a drifted output_schema. serialized_spec + manifest are deep-frozen
        so the caller mutating its own dict after prepare cannot affect the
        persisted snapshot.

        ``model_bundle`` is the resolved model bundle (available when prepare
        runs after compile); its revision is recorded in the manifest so resume
        can refuse on provider drift. Optional so callers that prepare before
        resolving the model (swarm members, tests) still work -- the manifest
        then records no provider revision."""
        fingerprint = spec_fingerprint(spec)
        snapshot = RunDefinitionSnapshot(
            run_id=context.run_id,
            runnable_type=str(context.runnable_type.value),
            runnable_id=context.runnable_id,
            serialized_spec=freeze_value(serialize_agent_spec(spec)),
            spec_fingerprint=fingerprint,
            user_id=context.user_id,
            tenant_id=context.tenant_id,
            workspace=context.workspace,
            provider_revision=None,
            created_at=datetime.now(timezone.utc),
            manifest=_agent_manifest(
                spec,
                str(context.runnable_type.value),
                fingerprint,
                model_bundle=model_bundle,
            ),
            resumability=Resumability.RESUMABLE.value,
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
            manifest=_agent_manifest(spec, str(context.runnable_type.value), swarm_fp),
            resumability=Resumability.RESUMABLE.value,
        )
        await self._run_definitions.create(snapshot)
        return snapshot


def _agent_manifest(
    spec: Any,
    runnable_type: str,
    fingerprint: str,
    *,
    model_bundle: Any = None,
) -> "Mapping[str, Any]":
    """Serialize the ExecutionManifest built from the spec-level inputs (the
    runnable id/type/fingerprint, the declared model name, and each tool
    declaration's descriptor fingerprint) plus the resolved model bundle's
    revision when available. Frozen so it cannot drift after the snapshot is
    taken. Compiled-handler revisions are layered in from the compiled run in a
    follow-up (tool handlers are currently resolved at execution time)."""
    manifest = build_execution_manifest(
        spec,
        runnable_type=runnable_type,
        runnable_fingerprint=fingerprint,
        model_provider=model_bundle,
    )
    return freeze_value(manifest_to_dict(manifest))
