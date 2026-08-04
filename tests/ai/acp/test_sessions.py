#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from types import SimpleNamespace

import pytest

from linktools.ai.acp.persistence import AcpSessionRepository
from linktools.ai.acp.session_paths import validate_session_paths
from linktools.ai.acp.sessions import AcpSessionService
from linktools.ai.execution.domain import RunStatus
from linktools.ai.governance.identity import trusted_local_principal


class _Runtime:
    def __init__(self) -> None:
        self.records = {}

    async def create_session(self, session_id, *, principal):
        return SimpleNamespace(id=session_id)

    async def get_execution_record(self, execution_id, *, principal):
        return self.records.get(execution_id)

    async def cancel(self, execution_id, *, principal):
        self.records[execution_id] = SimpleNamespace(status=RunStatus.CANCELLED)

    async def get_session_messages(self, *, session_id, principal):
        return ()


def test_session_paths_allow_project_children_and_reject_escape(tmp_path) -> None:
    project = tmp_path / "project"
    child = project / "child"
    outside = tmp_path / "outside"
    child.mkdir(parents=True)
    outside.mkdir()

    cwd, additional = validate_session_paths(
        project_root=project,
        cwd=str(child),
        additional_directories=[str(child), str(child)],
    )
    assert cwd == str(child.resolve())
    assert additional == (str(child.resolve()),)

    with pytest.raises(Exception):
        validate_session_paths(project_root=project, cwd=str(outside), additional_directories=[])


@pytest.mark.asyncio
async def test_load_replaces_additional_directories(tmp_path) -> None:
    project = tmp_path / "project"
    first = project / "first"
    second = project / "second"
    project.mkdir()
    first.mkdir()
    second.mkdir()
    runtime = _Runtime()
    service = AcpSessionService(
        runtime=runtime,
        repository=AcpSessionRepository(tmp_path / "state"),
        project_root=project,
        principal=trusted_local_principal(),
        default_mode_id="default",
    )
    active = await service.create(cwd=str(project), additional_directories=[str(first), str(second)])

    loaded = await service.load_or_resume(
        session_id=active.record.session_id,
        cwd=str(project),
        additional_directories=[str(second)],
        mcp_servers=[],
        replay=False,
    )

    assert loaded.record.additional_directories == (str(second.resolve()),)
    assert service.repository.load(active.record.session_id).additional_directories == (
        str(second.resolve()),
    )


@pytest.mark.asyncio
async def test_close_failure_is_retryable_and_not_persisted(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    runtime = _Runtime()
    attempts = 0

    class ClientServices:
        async def close_session_resources(self, active):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return (("terminal", "terminal-1", RuntimeError("release failed")),)
            active.terminal_handles.clear()
            active.pending_elicitation_ids.clear()
            return ()

    service = AcpSessionService(
        runtime=runtime,
        repository=AcpSessionRepository(tmp_path / "state"),
        project_root=project,
        principal=trusted_local_principal(),
        default_mode_id="default",
        client_services=ClientServices(),
    )
    active = await service.create(cwd=str(project))
    active.terminal_handles.add("terminal-1")

    first = await service.close_session_resources(active.record.session_id, reason="client")
    assert first.closed is False
    assert service.repository.load(active.record.session_id).closed is False
    assert active.closing is True

    active.terminal_handles.add("terminal-1")
    second = await service.close_session_resources(active.record.session_id, reason="client")
    assert second.closed is True
    assert service.repository.load(active.record.session_id).closed is True
