import asyncio
from pathlib import Path

import pytest

from linktools.ai.execution import local as local_module
from linktools.ai.execution.local import LocalExecutionBackend
from linktools.ai.execution.protocols import ExecutionBackend


@pytest.fixture()
def backend(tmp_path: Path) -> LocalExecutionBackend:
    return LocalExecutionBackend(runtime_dir=tmp_path, base_dirs=[])


def test_local_backend_satisfies_protocol(backend):
    assert isinstance(backend, ExecutionBackend)


def test_write_then_read_file_roundtrip(backend, tmp_path):
    write = asyncio.run(backend.write_file("out/data.txt", content="hello"))
    assert "error" not in write
    assert (tmp_path / "out" / "data.txt").read_text(encoding="utf-8") == "hello"
    read = asyncio.run(backend.read_file("out/data.txt"))
    assert read.get("content") == "hello" or "hello" in str(read)


def test_write_file_rejects_path_escape(backend):
    result = asyncio.run(backend.write_file("../escape.txt", content="x"))
    assert "error" in result


def test_run_bash_executes_in_runtime_dir(backend, tmp_path):
    result = asyncio.run(backend.run_bash("pwd"))
    assert result["exit_code"] == 0
    assert str(tmp_path) in result["stdout"]


def test_run_bash_timeout(backend):
    result = asyncio.run(backend.run_bash("sleep 5", timeout_ms=300))
    assert "error" in result and "timeout" in result["error"]


def test_terminate_kills_in_flight_subprocess(backend):
    async def main():
        task = asyncio.ensure_future(backend.run_bash("sleep 30"))
        # Poll until the subprocess is tracked, so terminate() has a live proc
        # to kill rather than racing the spawn.
        for _ in range(200):
            if backend._subprocesses:
                break
            await asyncio.sleep(0.01)
        assert backend._subprocesses, "run_bash did not register its subprocess"
        await backend.terminate()
        result = await task
        assert not backend._subprocesses, "registry was not cleared after terminate()"
        return result

    result = asyncio.run(main())
    # SIGKILL from terminate() surfaces as a negative (signal) exit code, and
    # the run does NOT return a timeout error -- the proc was reaped mid-flight.
    assert "error" not in result
    assert isinstance(result.get("exit_code"), int) and result["exit_code"] < 0


def test_fork_copies_runtime_dir_and_isolates_writes(backend, tmp_path):
    (tmp_path / "shared.txt").write_text("original")
    branch_dir = tmp_path.parent / "branch1"

    branch = asyncio.run(backend.fork(branch_dir))

    assert branch.runtime_dir == branch_dir
    assert (branch_dir / "shared.txt").read_text() == "original"

    asyncio.run(branch.write_file("shared.txt", content="modified-in-branch"))
    parent_read = asyncio.run(backend.read_file("shared.txt"))
    branch_read = asyncio.run(branch.read_file("shared.txt"))
    assert parent_read["content"] == "original"
    assert branch_read["content"] == "modified-in-branch"


def test_fork_result_satisfies_protocol(backend, tmp_path):
    branch = asyncio.run(backend.fork(tmp_path.parent / "branch2"))
    assert isinstance(branch, ExecutionBackend)


def test_apply_patch_modifies_file(backend, tmp_path):
    (tmp_path / "foo.txt").write_text("line1\nline2\n")
    diff = (
        "--- a/foo.txt\n"
        "+++ b/foo.txt\n"
        "@@ -1,2 +1,2 @@\n"
        " line1\n"
        "-line2\n"
        "+line2-patched\n"
    )
    result = asyncio.run(backend.apply_patch(diff))
    assert result.get("ok") is True
    assert (tmp_path / "foo.txt").read_text() == "line1\nline2-patched\n"


def test_apply_patch_rejects_path_escape(backend, tmp_path):
    diff = (
        "--- a/../../../etc/passwd\n"
        "+++ b/../../../etc/passwd\n"
        "@@ -1,1 +1,1 @@\n"
        "-x\n"
        "+y\n"
    )
    result = asyncio.run(backend.apply_patch(diff))
    assert "error" in result
    assert not Path("/etc/passwd").read_text().startswith("y")


def test_apply_patch_reports_failed_hunk(backend, tmp_path):
    (tmp_path / "foo.txt").write_text("line1\nline2\n")
    diff = "--- a/foo.txt\n+++ b/foo.txt\n@@ -1,2 +1,2 @@\n line1\n-nomatch\n+xxx\n"
    result = asyncio.run(backend.apply_patch(diff))
    assert "error" in result
    assert (tmp_path / "foo.txt").read_text() == "line1\nline2\n"


def test_apply_patch_rejects_empty_diff(backend):
    result = asyncio.run(backend.apply_patch(""))
    assert "error" in result


def test_apply_patch_rejects_non_ab_prefix_targeting_forbidden_path(backend, tmp_path):
    # Reviewer-reported bypass: a non-git first path component (here "zzz")
    # is not "a"/"b", so the old _strip_patch_prefix left it unstripped as
    # "zzz/capabilities/x.txt", which does not start with the forbidden
    # "capabilities/" prefix and so passed validation. But real `patch -p1`
    # unconditionally strips the first component and writes to
    # runtime_dir/capabilities/x.txt, bypassing the denylist entirely.
    (tmp_path / "capabilities").mkdir()
    (tmp_path / "capabilities" / "x.txt").write_text("orig\n")
    diff = (
        "--- zzz/capabilities/x.txt\n"
        "+++ zzz/capabilities/x.txt\n"
        "@@ -1 +1 @@\n"
        "-orig\n"
        "+pwned\n"
    )
    result = asyncio.run(backend.apply_patch(diff))
    assert "error" in result
    assert (tmp_path / "capabilities" / "x.txt").read_text() == "orig\n"


def test_apply_patch_timeout(backend, tmp_path, monkeypatch):
    # apply_patch has no caller-configurable timeout (unlike run_bash's
    # timeout_ms), so the module-level APPLY_PATCH_TIMEOUT_S constant is
    # dropped to 0 to deterministically force the real asyncio.wait_for
    # timeout path to fire against a genuine `patch` subprocess call --
    # not mocked. A timeout of 0 always expires before the subprocess can
    # finish (even a no-op process), so this is race-free regardless of
    # machine speed, unlike a small-but-nonzero timeout.
    monkeypatch.setattr(local_module, "APPLY_PATCH_TIMEOUT_S", 0)
    (tmp_path / "foo.txt").write_text("line1\nline2\n")
    diff = (
        "--- a/foo.txt\n"
        "+++ b/foo.txt\n"
        "@@ -1,2 +1,2 @@\n"
        " line1\n"
        "-line2\n"
        "+line2-patched\n"
    )
    result = asyncio.run(backend.apply_patch(diff))
    assert "error" in result and "timeout" in result["error"]
    # Original file must be untouched -- patch process was killed mid-flight.
    assert (tmp_path / "foo.txt").read_text() == "line1\nline2\n"


def test_apply_patch_rejects_arbitrary_first_component_path_escape(backend, tmp_path):
    # A non-git first path component ("src") combined with a path-escape
    # target must still be rejected once _strip_patch_prefix strips "src/"
    # (matching real patch -p1 semantics) and _resolve_runtime_path sees the
    # traversal.
    diff = (
        "--- src/../../../etc/passwd\n"
        "+++ src/../../../etc/passwd\n"
        "@@ -1,1 +1,1 @@\n"
        "-x\n"
        "+y\n"
    )
    result = asyncio.run(backend.apply_patch(diff))
    assert "error" in result
    assert not Path("/etc/passwd").read_text().startswith("y")
