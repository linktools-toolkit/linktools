import json
import os
from datetime import datetime, timezone

from linktools.ai.acp.persistence import (
    AcpSessionRecord,
    AcpSessionRepository,
    ProjectProcessLock,
    mcp_descriptor_fingerprint,
)


def test_sidecar_is_private_atomic_and_secret_free(tmp_path) -> None:
    repository = AcpSessionRepository(tmp_path)
    record = AcpSessionRecord(
        schema_version=1,
        session_id="session-1",
        cwd=str(tmp_path),
        additional_directories=(),
        mode_id="default",
        config_values={},
        mcp_server_fingerprints=(mcp_descriptor_fingerprint({"url": "https://example"}),),
        title=None,
        closed=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    repository.save(record)

    path = repository.path_for(record.session_id)
    assert os.stat(path).st_mode & 0o777 == 0o600
    raw = path.read_text()
    assert "secret-token" not in raw
    assert json.loads(raw)["schema_version"] == 1
    assert repository.load(record.session_id) == record


def test_project_lock_rejects_second_holder(tmp_path) -> None:
    first = ProjectProcessLock(tmp_path / "agent.lock")
    second = ProjectProcessLock(tmp_path / "agent.lock")
    first.acquire(project_root=tmp_path)
    try:
        try:
            second.acquire(project_root=tmp_path)
        except RuntimeError as exc:
            assert "held" in str(exc)
        else:
            raise AssertionError("second process lock unexpectedly acquired")
    finally:
        first.release()
