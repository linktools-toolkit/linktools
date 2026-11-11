#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable Trace snapshots preserve events across physical storage backends."""

from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.persistence._step_contracts import StepEvent
from linktools.ai.persistence.snapshots import RuntimeSnapshotOperations, SnapshotLimits, SnapshotResources, restore_snapshot
from linktools.ai.storage import InMemoryObjectStore

from .test_step_archive_read_boundaries import _archive


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_trace_snapshot_restores_event_owner_and_sequence(tmp_path: Path, backend: str) -> None:
    limits = SnapshotLimits(1000, 2**20)
    transport = InMemoryObjectStore("portable")
    objects = InMemoryObjectStore("runtime")

    def resources(store, *, read_only: bool) -> SnapshotResources:
        return SnapshotResources(
            namespace="archive-boundaries", tenant_id="tenant", durable_domains=frozenset({PersistenceDomain.TRACE}),
            stores={PersistenceDomain.TRACE: store}, object_stores={domain: objects for domain in PersistenceDomain},
            read_only=read_only,
        )

    async with _archive(tmp_path, backend) as (_state, archive, run):
        await archive.register_agent_run(run, execution_id="execution")
        events = [StepEvent("run", "MODEL_REQUEST_STARTED", index,
                            metadata={"linktools.ai.model_request_seq": str(index)}) for index in range(1, 4)]
        for value in events:
            await archive.append_event(value, execution_id="execution")
        source = resources(archive.state_store, read_only=True)
        operations = RuntimeSnapshotOperations(source, require_ready=lambda: None)
        with pytest.raises(AIError) as limited:
            await operations.export_snapshot(object_store=transport, limits=SnapshotLimits(2, 2**20))
        assert limited.value.code is ErrorCode.SNAPSHOT_UNSUPPORTED
        reference = await operations.export_snapshot(object_store=transport, limits=limits)

    storage = PersistenceStorage.sqlite(tmp_path / "restored.sqlite")
    async with Persistence.open("archive-boundaries", tenant_id="tenant", storage=storage) as target:
        restored = target.run_store.read_store(PersistenceDomain.TRACE)

        @asynccontextmanager
        async def open_target(root, namespace, tenant_id):
            yield resources(restored.state_store, read_only=False)

        await restore_snapshot(reference, object_store=transport, root=tmp_path / "restored",
                               limits=limits, open_target=open_target)
        assert await restored.list_events(agent_run_id="run") == events
        assert not any(value.kind == "step_event" for value in await restored.state_store.read(lambda tx: tx.scan_facts()))
        assert not any(value.kind == "history_association" for value in await restored.state_store.read(lambda tx: tx.scan_records()))
        roundtrip = await RuntimeSnapshotOperations(resources(restored.state_store, read_only=True),
                                                    require_ready=lambda: None).export_snapshot(
            object_store=transport, limits=limits,
        )
        assert roundtrip == reference
