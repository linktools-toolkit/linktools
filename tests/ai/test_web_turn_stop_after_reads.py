#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A Stop acknowledges the durable intent while model cancellation drains."""

import asyncio
from pathlib import Path

import httpx
import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart

from linktools.ai.capability import SessionCapability
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.turns import SessionTurnKey
from linktools.ai.runtime import Runtime
from linktools.ai.workspace import Workspace

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features, workspace_features
from .test_canonical_turn_subagents import _HEADERS, _SESSION, _HostAuthorization, _events


@pytest.mark.asyncio
@pytest.mark.parametrize("first_request_interrupted", (False, True))
async def test_stop_after_file_reads_publishes_cancelling_and_accepts_a_second_request(
    tmp_path: Path, first_request_interrupted: bool,
) -> None:
    pytest.importorskip("starlette")
    from linktools.ai.web import create_app

    for name in ("a", "b", "c"):
        (tmp_path / f"{name}.txt").write_text(f"fixture {name}", encoding="utf-8")
    reads_finished = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def model(messages, _info):
        results = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
        if not results:
            return ModelResponse(parts=[
                ToolCallPart("read_file", {"path": f"{name}.txt"}, tool_call_id=f"read-{name}")
                for name in ("a", "b", "c")
            ])
        assert {part.tool_call_id for part in results if part.outcome == "success"} == {
            "read-a", "read-b", "read-c",
        }
        reads_finished.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cleanup_started.set()
            await release_cleanup.wait()
            raise

    name = "web-stop-after-reads"
    features = workspace_features(name, workspace=Workspace.load(tmp_path))
    features.capability(SessionCapability(checkpoints=True), scope="runtime")
    features.agent("default", model="default", allow_tools=("read_file",))
    model_features(_Models(_UsageFunctionModel(model)), features)
    async with (
        Persistence.open(name, storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as state,
        Runtime.open(persistence=state, features=features, authorization=_HostAuthorization()) as runtime,
    ):
        app = create_app(runtime=runtime, local_workspace=True)
        params = {"session_id": _SESSION}
        async with (
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765") as http,
            asyncio.timeout(10),
        ):
            created = await http.post("/api/sessions", headers=_HEADERS,
                                      json={"session_id": _SESSION, "request_id": "create"})
            assert created.status_code == 201, created.text
            async with _events(app) as frames:
                assert (await frames.get())["type"] == "snapshot"
                response = await http.post("/api/session/messages", params=params, headers=_HEADERS,
                                           json={"prompt": "Read three files", "request_id": "read-three"})
                assert response.status_code == 202, response.text
                turn_id = response.json()["turn_id"]
                await reads_finished.wait()

                async def stop(request_id):
                    result = await http.post("/api/session/cancel", params=params, headers=_HEADERS,
                                             json={"request_id": request_id, "turn_id": turn_id})
                    assert result.status_code == 200 and result.json() == {"accepted": True}, result.text

                async def next_frame(kind, phase=None):
                    while True:
                        frame = await frames.get()
                        if frame.get("turn_id") == turn_id and frame["type"] == kind and frame.get("phase") == phase:
                            return frame

                try:
                    if first_request_interrupted:
                        # The first HTTP handler did not reach the peer signal.
                        assert await state.session_turns.request_cancel(SessionTurnKey(_SESSION, turn_id))
                    else:
                        await stop("first")
                    await stop("second")
                    await cleanup_started.wait()
                    await next_frame("gap", "cancelling")
                    await stop("while-draining")
                finally:
                    release_cleanup.set()
                assert (await next_frame("terminal"))["state"] == "cancelled"
                await stop("after-cancelled")
