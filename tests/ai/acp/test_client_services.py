#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from linktools.ai.acp.client_services import AcpClientServices
from linktools.ai.acp.persistence import AcpSessionRecord
from linktools.ai.acp.session_models import ActiveAcpSession
from linktools.ai.acp.session_state import SessionOperationCoordinator, SessionOperationKind


def _session(tmp_path):
    record = AcpSessionRecord(
        schema_version=1,
        session_id="s1",
        cwd=str(tmp_path),
        additional_directories=(),
        mode_id="default",
        config_values={},
        mcp_server_fingerprints=(),
        title=None,
        closed=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    return ActiveAcpSession(record, asyncio.Lock(), None, SimpleNamespace(), set(), set())


@pytest.mark.asyncio
async def test_client_file_service_enforces_capability_and_root(tmp_path) -> None:
    session = _session(tmp_path)
    services = AcpClientServices(project_root=tmp_path)
    class Connection:
        async def read_text_file(self, session_id, path, **kwargs):
            assert session_id == "s1"
            assert path == str(tmp_path / "file.txt")
            return SimpleNamespace(content="client content")

    services.set_connection(Connection(), SimpleNamespace(fs=SimpleNamespace(read_text_file=True)))

    response = await services.read_text_file(session, "file.txt")
    assert response.content == "client content"
    with pytest.raises(Exception):
        await services.read_text_file(session, "../outside.txt")


@pytest.mark.asyncio
async def test_client_file_write_uses_remote_callback(tmp_path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("local content", encoding="utf-8")
    session = _session(tmp_path)
    calls = []

    class Connection:
        async def write_text_file(self, session_id, path, content):
            calls.append((session_id, path, content))
            return SimpleNamespace()

    services = AcpClientServices(project_root=tmp_path)
    services.set_connection(
        Connection(),
        SimpleNamespace(fs=SimpleNamespace(write_text_file=True)),
    )

    await services.write_text_file(session, "file.txt", "remote content")

    assert calls == [("s1", str(target), "remote content")]
    assert target.read_text(encoding="utf-8") == "local content"


@pytest.mark.asyncio
async def test_client_file_capability_is_required(tmp_path) -> None:
    services = AcpClientServices(project_root=tmp_path)
    services.set_connection(object(), SimpleNamespace(fs=SimpleNamespace()))

    with pytest.raises(Exception):
        await services.read_text_file(_session(tmp_path), "file.txt")


@pytest.mark.asyncio
async def test_terminal_create_after_prompt_cancel_is_compensated(tmp_path) -> None:
    session = _session(tmp_path)
    session.active_execution_id = "execution-1"
    coordinator = SessionOperationCoordinator()
    operation = await coordinator.reserve(
        session,
        SessionOperationKind.PROMPT,
        execution_id="execution-1",
    )
    started = asyncio.Event()
    release = asyncio.Event()
    calls = []

    class Connection:
        async def create_terminal(self, session_id, **kwargs):
            started.set()
            await release.wait()
            return SimpleNamespace(terminal_id="terminal-1")

        async def kill_terminal(self, session_id, terminal_id):
            calls.append(("kill", terminal_id))

        async def release_terminal(self, session_id, terminal_id):
            calls.append(("release", terminal_id))

    services = AcpClientServices(project_root=tmp_path)
    services.set_connection(Connection(), SimpleNamespace(terminal=True))
    task = asyncio.create_task(services.create_terminal(session))
    await started.wait()
    session.active_execution_id = None
    release.set()

    with pytest.raises(Exception):
        await task
    assert calls == [("kill", "terminal-1"), ("release", "terminal-1")]
    assert not session.terminal_handles
    await coordinator.release(session, operation)
