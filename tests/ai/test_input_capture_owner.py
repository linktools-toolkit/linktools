#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Input capture composition borrows persistence without constructing Runtime."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import BinaryContent

from linktools.ai.core import Principal, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.inputs import InputCaptures
from linktools.ai.migrate import provision_database
from linktools.ai.persistence import Persistence, PersistenceStorage, PersistenceDomain
from linktools.ai.persistence.captures import input_capture_key, read_capture_record, capture_record_value, capture_retention_value
from linktools.ai.persistence.codec import iter_runtime_object_refs
from linktools.ai.storage import FilesystemObjectStore, read_object
from linktools.ai.task import TaskGraphTemplate, TaskNode, TaskRef


def test_input_capture_owner_import_is_independent() -> None:
    subprocess.run([sys.executable, "-c", (
        "import sys; import linktools.ai.inputs; "
        "assert not any(name == 'linktools.ai.runtime' or name.startswith('linktools.ai.runtime.') "
        "for name in sys.modules)"
    )], check=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "local_sqlite", "sqlite", "sql"))
async def test_input_capture_owner_borrows_storage_and_roundtrips_v1(
    tmp_path: Path, backend: str,
) -> None:
    engine = None
    if backend == "memory":
        storage = PersistenceStorage.in_memory()
    elif backend == "local_sqlite":
        storage = PersistenceStorage.from_root(tmp_path / "data", exclusive_writer=True)
    elif backend == "sqlite":
        storage = PersistenceStorage.sqlite(tmp_path / "data.sqlite")
    else:
        from sqlalchemy.ext.asyncio import create_async_engine
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'sql.sqlite'}")
        await provision_database(engine)
        storage = PersistenceStorage.sql(engine, object_store=FilesystemObjectStore(tmp_path / "objects"))
    principal = Principal("owner", "tenant", "service")
    policy = TenantAuthorizationPolicy("tenant")
    try:
        async with Persistence.open("captures", storage=storage, tenant_id="tenant") as persistence:
            captures = InputCaptures.from_persistence(
                persistence, authorization=policy, execution=SimpleNamespace(),
            )
            image = BinaryContent(data=b"image bytes", media_type="image/png")
            reference = await captures.create_agent_input(
                ("look", image), principal=principal, idempotency_key="input",
            )
            # A second producer uses the same immutable captures without the first producer.
            reader = InputCaptures.from_persistence(
                persistence, authorization=policy, execution=SimpleNamespace(),
            )
            value = await reader.read_agent(reference, principal=principal)
            assert value.prompt[0] == "look"
            assert any(isinstance(item, BinaryContent) and item.data == image.data for item in value.prompt)
            assert value.source_execution_id is None
            objects = persistence.object_store(PersistenceDomain.TASK)
            key = input_capture_key("captures", "tenant", "agent", reference.capture_id)
            assert await objects.stat(key) is None
            record = await read_capture_record(
                persistence.task.tasks.state_store, namespace="captures", tenant_id="tenant", key=key,
            )
            assert record is not None
            payload = capture_record_value(record)
            refs = tuple(iter_runtime_object_refs(capture_retention_value(record), default_domain=PersistenceDomain.TASK))
            assert len(refs) == 1
            domain, attachment = refs[0]
            assert domain is PersistenceDomain.TASK
            assert await read_object(objects, attachment.key, expected_digest=attachment.digest,
                                     expected_size=attachment.size) == image.data
            assert payload["version"] == 1
            template = TaskGraphTemplate((TaskNode("node", task=TaskRef("task", 1)),))
            graph = await captures.create_graph_template(template, principal=principal, idempotency_key="graph")
            assert await reader.read_graph(graph, principal=principal) == template
            with pytest.raises(AIError) as denied:
                await reader.read_agent(reference, principal=Principal("other", "tenant", "service"))
            assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
            assert persistence.ready
    finally:
        if engine is not None:
            await engine.dispose()
