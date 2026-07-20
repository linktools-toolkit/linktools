#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""SEC-01 (v5 guide §5): file reads must follow symlinks to their real target
and reject any whose resolved path leaves every allowed root.

A symlink that *lives* inside ``runtime/`` but points at ``/etc/passwd`` (or any
file/dir outside the allowed roots) must be denied -- the lexical position of
the link is irrelevant. Internal symlinks (real target still inside a root)
stay usable. Each test fails before the fix (the link is followed and the
secret is returned)."""

import asyncio

import pytest

from linktools.ai.execution.local import LocalExecutionBackend, _run_file_tool_sync


def _run(tool: str, args: dict, runtime) -> dict:
    return _run_file_tool_sync(tool, args, runtime, [])


def test_read_file_rejects_symlink_to_outside_file(tmp_path):
    runtime = tmp_path / "runtime"
    outside = tmp_path / "outside"
    runtime.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("TOP-SECRET")
    (runtime / "link.txt").symlink_to(outside / "secret.txt")

    result = _run("read_file", {"path": "runtime/link.txt"}, runtime)

    assert "error" in result, result
    assert "TOP-SECRET" not in str(result)


def test_read_files_rejects_symlink_to_outside_file(tmp_path):
    runtime = tmp_path / "runtime"
    outside = tmp_path / "outside"
    runtime.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("TOP-SECRET")
    (runtime / "link.txt").symlink_to(outside / "secret.txt")

    result = _run("read_files", {"paths": ["runtime/link.txt"]}, runtime)

    assert "files" in result
    assert result["files"][0].get("error"), result
    assert "TOP-SECRET" not in str(result)


def test_read_json_rejects_symlink_to_outside_file(tmp_path):
    runtime = tmp_path / "runtime"
    outside = tmp_path / "outside"
    runtime.mkdir()
    outside.mkdir()
    (outside / "data.json").write_text('{"secret": "TOP-SECRET"}')
    (runtime / "link.json").symlink_to(outside / "data.json")

    result = _run("read_json", {"path": "runtime/link.json"}, runtime)

    assert "error" in result, result
    assert "TOP-SECRET" not in str(result)


def test_read_jsons_rejects_symlink_to_outside_file(tmp_path):
    runtime = tmp_path / "runtime"
    outside = tmp_path / "outside"
    runtime.mkdir()
    outside.mkdir()
    (outside / "data.json").write_text('{"secret": "TOP-SECRET"}')
    (runtime / "link.json").symlink_to(outside / "data.json")

    result = _run("read_jsons", {"files": [{"path": "runtime/link.json"}]}, runtime)

    assert result["files"][0].get("error"), result
    assert "TOP-SECRET" not in str(result)


def test_list_dir_rejects_symlink_to_outside_dir(tmp_path):
    runtime = tmp_path / "runtime"
    outside = tmp_path / "outside"
    runtime.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("TOP-SECRET")
    (runtime / "link_dir").symlink_to(outside)

    result = _run("list_dir", {"path": "runtime/link_dir"}, runtime)

    assert "error" in result, result
    assert "TOP-SECRET" not in str(result)


def test_batch_files_read_rejects_symlink_to_outside_file(tmp_path):
    runtime = tmp_path / "runtime"
    outside = tmp_path / "outside"
    runtime.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("TOP-SECRET")
    (runtime / "link.txt").symlink_to(outside / "secret.txt")

    result = _run(
        "batch_files",
        {"operations": [{"action": "read", "path": "runtime/link.txt"}]},
        runtime,
    )

    assert result["operations"][0].get("error"), result
    assert "TOP-SECRET" not in str(result)


def test_internal_symlink_whose_target_stays_in_root_is_allowed(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "real.txt").write_text("ok-content")
    (runtime / "link.txt").symlink_to(runtime / "real.txt")

    result = _run("read_file", {"path": "runtime/link.txt"}, runtime)

    assert result.get("content") == "ok-content", result


def test_write_file_rejects_parent_symlink_to_outside_dir(tmp_path):
    runtime = tmp_path / "runtime"
    outside = tmp_path / "outside"
    runtime.mkdir()
    outside.mkdir()
    (runtime / "out").symlink_to(outside)

    backend = LocalExecutionBackend(runtime_dir=runtime, base_dirs=[])
    result = asyncio.run(backend.write_file("out/secret.txt", content="x"))

    assert "error" in result, result
    assert not (outside / "secret.txt").exists()


@pytest.mark.parametrize("rel", ["link.txt", "runtime/link.txt"])
def test_symlink_escape_rejected_with_or_without_prefix(tmp_path, rel):
    runtime = tmp_path / "runtime"
    outside = tmp_path / "outside"
    runtime.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("TOP-SECRET")
    (runtime / "link.txt").symlink_to(outside / "secret.txt")

    result = _run("read_file", {"path": rel}, runtime)

    assert "error" in result, result
    assert "TOP-SECRET" not in str(result)
