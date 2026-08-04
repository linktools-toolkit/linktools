import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from linktools.ai.acp.client_services import AcpClientServices
from linktools.ai.acp.persistence import AcpSessionRecord
from linktools.ai.acp.sessions import ActiveAcpSession


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
    target = tmp_path / "file.txt"
    target.write_text("hello", encoding="utf-8")
    session = _session(tmp_path)
    services = AcpClientServices(project_root=tmp_path)
    services.set_connection(None, SimpleNamespace(fs=SimpleNamespace(read_text_file=True)))

    response = await services.read_text_file(session, "file.txt")
    assert response.content == "hello"
    with pytest.raises(Exception):
        await services.read_text_file(session, "../outside.txt")
