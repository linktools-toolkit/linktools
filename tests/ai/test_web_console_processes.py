#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Web reads and explicit controls share the cross-process Runtime owner."""

import asyncio
from pathlib import Path

import httpx
import pytest

pytest.importorskip("starlette")
pytest.importorskip("uvicorn")

from pydantic_ai.models.function import FunctionModel

from linktools.ai.runtime import Runtime, RuntimeHistory, RuntimeStorage
from linktools.ai.web import create_app

from ._runtime_test_helpers import _wait_for_committed
from .test_live_history_readback_integration import _Models
from .test_runtime_recovery_ownership import _NAMESPACE, _RuntimeProcess, _capabilities
from .test_web_console import _HEADERS, client


async def _get(http: httpx.AsyncClient, path: str) -> dict[str, object]:
    response = await http.get(path)
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.asyncio
async def test_web_reads_live_process_history_without_taking_ownership(tmp_path: Path) -> None:
    database = tmp_path / "runtime.db"
    owner = await asyncio.to_thread(_RuntimeProcess, database)
    identity = owner.initial["execution_id"]
    calls = []

    async def model(messages, info):
        del messages, info
        calls.append("unexpected replay")
        yield "unexpected replay"

    try:
        async with Runtime.open(
            _NAMESPACE, models=_Models(FunctionModel(stream_function=model)),
            storage=RuntimeStorage.sqlite(database), capabilities=(_capabilities(),),
        ) as runtime, RuntimeHistory.open(
            _NAMESPACE, storage=RuntimeStorage.sqlite(database),
        ) as history, client(create_app(runtime=runtime)) as writable, client(create_app(history=history)) as readonly:
            for http in (writable, readonly):
                page = await _wait_for_committed(
                    lambda: _get(http, f"/api/executions/{identity}/models?include_content=true"),
                    lambda value: bool(value["items"]), timeout=10,
                )
                request = page["items"][0]
                assert request["execution_id"] == identity and request["model_request_seq"] == 1
                assert request["status"] == "RUNNING"
                assert request["request"] == {} and request["response"] is None
                assert request["usage"] is None and request["duration_ns"] is None
                content = await _get(http, f"/api/executions/{identity}/history?include_content=true")
                assert any(item["content"] == "Wait" for item in content["items"])
                for _ in range(2):
                    assert (await _get(http, f"/api/executions/{identity}"))["status"] == "STARTED"
                assert (await _get(http, "/api/session?session_id=session"))["session"]["active_execution_id"] == identity
            assert not calls
            assert (await asyncio.to_thread(owner.request, "finish"))["status"] == "SUCCEEDED"
            for http in (writable, readonly):
                result = await _get(http, f"/api/executions/{identity}/result")
                assert result["output"] == {"text": "owner answer"}
                models = await _get(http, f"/api/executions/{identity}/models?include_content=true")
                assert models["items"][0]["status"] == "SUCCEEDED"
                assert models["items"][0]["content_included"]
            assert not calls
    finally:
        await asyncio.to_thread(owner.close)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_first", (False, True))
async def test_web_recovery_requires_explicit_control_after_process_exit(
    tmp_path: Path, cancel_first: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "runtime.db"
    owner = await asyncio.to_thread(_RuntimeProcess, database)
    identity = owner.initial["execution_id"]
    calls = []

    async def model(messages, info):
        del messages, info
        calls.append("recovered")
        yield "recovered answer"

    try:
        if not cancel_first:
            await asyncio.to_thread(owner.crash)
        storage = RuntimeStorage.sqlite(database)
        async with Runtime.open(
            _NAMESPACE, models=_Models(FunctionModel(stream_function=model)),
            storage=storage, capabilities=(_capabilities(),),
        ) as runtime, client(create_app(runtime=runtime)) as http:
            if cancel_first:
                await _wait_for_committed(
                    lambda: _get(http, f"/api/executions/{identity}/models"),
                    lambda value: bool(value["items"]), timeout=10,
                )
                tenant = runtime.default_principal.tenant_id
                backend = runtime._execution_service.runtime_backend()
                commit_cancel = backend.commit_cancel_checkpoint
                advanced = False

                async def race_revision(commit, *, expected_status):
                    nonlocal advanced
                    if not advanced:
                        advanced = True
                        await storage.execution.executions.acquire_dependency_hold(
                            identity, tenant_id=tenant, hold_id="concurrent-reader",
                        )
                    return await commit_cancel(commit, expected_status=expected_status)

                # Schedule one real SQL write between the control read and its CAS.
                monkeypatch.setattr(backend, "commit_cancel_checkpoint", race_revision)
                path = f"/api/executions/{identity}/cancel"
                request = {"request_id": "cancel"}
                conflict = await http.post(path, json=request, headers=_HEADERS)
                assert conflict.status_code == 409, conflict.text
                assert conflict.json()["code"] == "STORAGE_CONFLICT"
                assert (await _get(http, f"/api/executions/{identity}"))["status"] == "STARTED"
                assert (await asyncio.to_thread(owner.request, "snapshot"))["status"] == "STARTED"
                events = await storage.execution.events.list(identity, tenant_id=tenant, after_event_seq=0, limit=100)
                assert not any(item.event_type == "CANCEL_REQUESTED" for item in events.items)
                await storage.execution.executions.release_dependency_hold(
                    identity, tenant_id=tenant, hold_id="concurrent-reader",
                )
                for _ in range(2):
                    cancelled = await http.post(path, json=request, headers=_HEADERS)
                    assert cancelled.status_code == 200, cancelled.text
                    assert cancelled.json()["cancelled"] is False
                events = await storage.execution.events.list(identity, tenant_id=tenant, after_event_seq=0, limit=100)
                assert sum(item.event_type == "CANCEL_REQUESTED" for item in events.items) == 1
                await asyncio.to_thread(owner.crash)
            for _ in range(2):
                info = await _get(http, f"/api/executions/{identity}")
                assert info["status"] == ("CANCELLING" if cancel_first else "STARTED")
            assert not calls
            recovered = await http.post(
                f"/api/executions/{identity}/recover", json={"request_id": "recover"}, headers=_HEADERS,
            )
            assert recovered.status_code == 202, recovered.text
            assert recovered.json()["execution_id"] == identity
            expected = "CANCELLED" if cancel_first else "SUCCEEDED"
            await _wait_for_committed(
                lambda: _get(http, f"/api/executions/{identity}"),
                lambda value: value["status"] == expected, timeout=10,
            )
            result = await _get(http, f"/api/executions/{identity}/result")
            assert result["status"] == expected
            assert result["output"] == (None if cancel_first else {"text": "recovered answer"})
            assert calls == ([] if cancel_first else ["recovered"])
    finally:
        await asyncio.to_thread(owner.close)
