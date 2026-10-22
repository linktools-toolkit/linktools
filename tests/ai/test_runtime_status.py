#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime status reports bound resources without opening or probing them."""
from ._runtime_test_helpers import model_features

import asyncio
from pathlib import Path

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.capability import TraceCapability
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.migrate import provision_database
from linktools.ai.observe import Metrics
from linktools.ai.runtime import Runtime, RuntimeContext, RuntimeStorage
from linktools.ai.storage import FilesystemObjectStore, StorageRoute
from ._runtime_test_helpers import RuntimeUsageModels


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "filesystem", "sqlite", "sql"])
async def test_runtime_status_reports_bound_storage_without_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    engine = None
    features = RuntimeFeatures("status")
    objects = FilesystemObjectStore(tmp_path / "custom-objects", store_id="chosen-objects")
    if backend == "memory":
        storage = RuntimeStorage.in_memory()
    elif backend == "filesystem":
        storage = RuntimeStorage.filesystem(tmp_path / "state", object_store=objects)
        features.capability(TraceCapability(storage=StorageRoute.filesystem(
            tmp_path / "state" / "custom-trace", transaction_root=tmp_path / "state",
        )))
    elif backend == "sqlite":
        storage = RuntimeStorage.sqlite(tmp_path / "chosen.db", object_store=objects)
    else:
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'private-connection.db'}")
        await provision_database(engine)
        storage = RuntimeStorage.sql(engine)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("status must not query, provision, or start background work")

    try:
        async with Runtime.open(
            "actual-namespace", context=RuntimeContext(None, tenant_id="actual-tenant"),
            storage=storage, features=model_features(RuntimeUsageModels(), features),
        ) as runtime:
            if engine is not None:
                event.listen(engine.sync_engine, "before_cursor_execute", forbidden)
            try:
                with monkeypatch.context() as probe:
                    probe.setattr(Path, "exists", forbidden)
                    probe.setattr(Path, "mkdir", forbidden)
                    probe.setattr(asyncio, "create_task", forbidden)
                    status = runtime.status()
                    assert runtime.status() == status
            finally:
                if engine is not None:
                    event.remove(engine.sync_engine, "before_cursor_execute", forbidden)
            assert status["namespace"] == "actual-namespace"
            assert status["tenant_id"] == "actual-tenant"
            assert status["metrics"].enabled is False
            assert status["metrics"].accepting is False
            routes = {item["domain"].value: item for item in status["storage"]}
            assert routes["execution"]["backend"] == backend
            assert routes["execution"]["retention"].value == ("volatile" if backend == "memory" else "durable")
            if backend in {"memory", "sql"}:
                assert all(item["path"] is None and item["object_store_paths"] == () for item in routes.values())
            else:
                expected_path = tmp_path / "state" / "execution" if backend == "filesystem" else tmp_path / "chosen.db"
                assert routes["execution"]["path"] == str(expected_path)
                assert routes["execution"]["object_store_id"] == "chosen-objects"
                assert routes["execution"]["object_store_paths"] == (str(tmp_path / "custom-objects"),)
            if backend == "filesystem":
                assert routes["trace"]["path"] == str(tmp_path / "state" / "custom-trace" / "trace")
            assert "private-connection" not in repr(status)
    finally:
        if engine is not None:
            await engine.dispose()


@pytest.mark.asyncio
async def test_web_config_reads_live_runtime_metrics() -> None:
    pytest.importorskip("starlette")
    pytest.importorskip("uvicorn")
    from linktools.ai.web import create_app
    from .test_web_console import client

    async with Runtime.open(
        "live-status", storage=RuntimeStorage.in_memory(),
        features=model_features(RuntimeUsageModels()), metrics=Metrics.in_memory(),
    ) as runtime, client(create_app(runtime=runtime)) as http:
        before = (await http.get("/api/config")).json()["runtime"]
        assert before["namespace"] == "live-status"
        assert before["metrics"]["enabled"] is True
        assert before["metrics"]["accepted"] == 0
        await (await runtime.agents.get().start("hello")).wait()
        await runtime.metrics.flush()
        after = (await http.get("/api/config")).json()["runtime"]
        assert after["metrics"]["accepted"] > before["metrics"]["accepted"]
        assert after["metrics"]["pending"] == 0
        assert after["storage"] == before["storage"]
