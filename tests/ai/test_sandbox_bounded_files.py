#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded previews, anchored discovery, and atomic streaming edit contracts."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.workspace import LocalSandbox, ReadOnlySandboxPolicy, SandboxOperationRejected
from linktools.ai.workspace import _file_io, _local_sandbox
from linktools.ai.workspace._bubblewrap import _read_workspace_bytes
from linktools.ai.workspace.sandbox_worker import _dispatch

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def isolated_content_detector_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_file_io, "_magic_available", None)
    monkeypatch.setattr(_file_io, "_magic_factory", None)
    monkeypatch.setattr(_file_io, "_magic_threads", _file_io._MagicThreadState())


@pytest.mark.parametrize("through_worker", [False, True])
async def test_preview_has_bounded_io_and_cursor_continues_long_unicode_line(
    tmp_path: Path, through_worker: bool,
) -> None:
    text = "é🙂" * 15000 + "\r\nsecond\nlast"
    (tmp_path / "long.log").write_text(text, encoding="utf-8", newline="")
    session = await LocalSandbox().open(root=tmp_path)
    async def read(**params: object) -> dict:
        if through_worker:
            return json.loads(await _dispatch(session, "read_text", {"path": "long.log", **params}))
        return dict(await session.read_text("long.log", **params))
    try:
        first = await read(max_bytes=4096)
        assert first["bytes_read"] <= 65536
        assert first["end_offset"] == 0
        assert first["end_line_offset"] > 0
        assert first["truncated"] and not first["eof"]
        assert "hash" not in first and "total_lines" not in first
        pieces = [first["content"]]
        current = first
        while current["next_cursor"]:
            current = await read(cursor=current["next_cursor"], max_bytes=4096)
            pieces.append(current["content"])
        assert "".join(pieces) == text
        assert current["eof"] and not current["truncated"]
    finally:
        await session.close()


async def test_requested_range_is_not_truncation_and_metadata_is_explicit(tmp_path: Path) -> None:
    raw = b"first\r\nsecond\rlast"
    (tmp_path / "data.txt").write_bytes(raw)
    session = await LocalSandbox().open(root=tmp_path)
    try:
        first = await session.read_text("data.txt", limit=1)
        assert first["content"] == "first\r\n"
        assert not first["truncated"] and not first["eof"]
        second = await session.read_text("data.txt", cursor=first["next_cursor"], limit=1)
        assert second["content"] == "second\r"
        metadata = await session.read_text("data.txt", offset=2, full_metadata=True)
        assert metadata["content"] == "last"
        assert metadata["hash"] == hashlib.sha256(raw).hexdigest()
        assert metadata["total_lines"] == 3
        assert metadata["bytes_read"] == len(raw) * 2
    finally:
        await session.close()


async def test_cursor_rejects_changed_source_wrong_path_and_offset(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("first\nsecond\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("first\nsecond\n", encoding="utf-8")
    session = await LocalSandbox().open(root=tmp_path)
    try:
        result = await session.read_text("a.txt", limit=1)
        for selected, path, offset in [(session, "b.txt", None), (session, "a.txt", 0)]:
            with pytest.raises(AIError) as error:
                await selected.read_text(path, cursor=result["next_cursor"], offset=offset)
            assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID
        (tmp_path / "a.txt").write_text("changed\nsecond\n", encoding="utf-8")
        with pytest.raises(AIError) as error:
            await session.read_text("a.txt", cursor=result["next_cursor"])
        assert error.value.code is ErrorCode.STORAGE_CONFLICT
    finally:
        await session.close()


async def test_lossy_preview_and_actual_read_budget_are_reported(tmp_path: Path) -> None:
    raw = b"a\xff\x00" + "🙂".encode() * 100
    (tmp_path / "log.txt").write_bytes(raw)
    session = await LocalSandbox().open(root=tmp_path)
    try:
        result = await session.read_text("log.txt", max_read_bytes=17)
        assert result["bytes_read"] == 17
        assert result["lossy"] and result["truncated"]
        pieces = [result["content"]]
        while result["next_cursor"]:
            result = await session.read_text("log.txt", cursor=result["next_cursor"], max_read_bytes=17)
            pieces.append(result["content"])
        assert "".join(pieces) == raw.decode("utf-8", "replace").replace("\x00", "\ufffd")
        with pytest.raises(AIError):
            await session.read_text("log.txt", full_metadata=True, max_read_bytes=17)
    finally:
        await session.close()


async def test_inspection_without_magic_preserves_basic_text_and_same_state_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "magic", None)
    (tmp_path / "activity.log").write_bytes(b"hello\xff\n")
    session = await LocalSandbox().open(root=tmp_path)
    try:
        facts = await session.inspect_file("activity.log")
        assert facts["is_text"]
        assert "detection_reason" not in facts
        assert facts["type_source"] in {"extension", "unknown"}
        assert facts["bytes_read"] == 7
        assert await session.read_bytes("activity.log", max_bytes=7, expected_state=facts["state"]) == b"hello\xff\n"
        assert _read_workspace_bytes(tmp_path, "activity.log", read_policy=None, hidden_paths=(), max_bytes=7, expected_state=facts["state"]) == b"hello\xff\n"
        (tmp_path / "activity.log").write_text("changed", encoding="utf-8")
        with pytest.raises(AIError) as error:
            await session.read_bytes("activity.log", expected_state=facts["state"])
        assert error.value.code is ErrorCode.STORAGE_CONFLICT
    finally:
        await session.close()


async def test_content_conflicts_and_container_types_are_not_guessed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = []
    class Detector:
        def __init__(self, *, mime: bool) -> None:
            assert mime
        def from_buffer(self, sample: bytes) -> str:
            seen.append(sample)
            return "application/zip"
    monkeypatch.setitem(sys.modules, "magic", SimpleNamespace(Magic=Detector))
    (tmp_path / "claim.png").write_bytes(b"PK" + b"\0" * 20000)
    session = await LocalSandbox().open(root=tmp_path)
    try:
        facts = await session.inspect_file("claim.png")
        assert facts["media_type"] == "application/zip"
        assert facts["type_source"] == "conflict" and not facts["is_text"]
        assert len(seen[0]) == 8192
        assert facts["bytes_read"] == 8192
    finally:
        await session.close()


async def test_list_and_search_share_path_anchored_glob_and_visibility(tmp_path: Path) -> None:
    folder = tmp_path / "src"
    (folder / "nested").mkdir(parents=True)
    for name in ("root.py", ".hidden.py", "nested/child.py", "nested/ignored.py", "text.txt"):
        (folder / name).write_text("needle\n", encoding="utf-8")
    policy = ReadOnlySandboxPolicy(("src/*.py", "src/nested/child.py",))
    session = await LocalSandbox(read_policy=policy).open(root=tmp_path)
    try:
        direct = await session.list_files("src", include_glob="*.py")
        recursive = await session.list_files("src", include_glob="**/*.py")
        search = await session.search_text("needle", path="src", include_glob="**/*.py")
        assert "root.py" in direct and ".hidden.py" in direct and "child.py" not in direct
        assert all(name in recursive and name in search for name in ("root.py", ".hidden.py", "child.py"))
        assert "ignored.py" not in recursive + search
        assert "child.py" not in await session.search_text("needle", path="src", include_glob="*.py")
        # Existing host search deliberately retains its prior workspace-relative fnmatch.
        assert "child.py" in await session.search_files("needle", path="src", include_glob="*.py")
    finally:
        await session.close()


async def test_search_long_lines_and_enumeration_limits_never_claim_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "long.txt").write_text("x" * 200000 + "needle\n", encoding="utf-8")
    session = await LocalSandbox().open(root=tmp_path)
    try:
        result = await session.search_text("needle")
        assert "coverage incomplete" in result and "No matches found" not in result
        monkeypatch.setattr(_local_sandbox, "_MAX_ENUMERATED_ITEMS", 1)
        (tmp_path / "another.txt").write_text("x", encoding="utf-8")
        assert "truncated" in await session.list_files()
    finally:
        await session.close()


async def test_streaming_edit_preserves_unrelated_changes_and_hash_chain(tmp_path: Path) -> None:
    text = "x" * (65536 - 2) + "old🙂value" + "y" * 70000
    path = tmp_path / "large.txt"
    path.write_text(text, encoding="utf-8")
    session = await LocalSandbox().open(root=tmp_path)
    try:
        old_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        path.write_text(text + "unrelated", encoding="utf-8")
        with pytest.raises(SandboxOperationRejected) as error:
            await session.edit_file("large.txt", "old🙂value", "new", expected_hash=old_hash)
        assert error.value.code is ErrorCode.STORAGE_CONFLICT
        result = await session.edit_file("large.txt", "old🙂value", "new")
        expected = text.replace("old🙂value", "new") + "unrelated"
        assert path.read_text(encoding="utf-8") == expected
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest in result
        await session.edit_file("large.txt", "new", "", expected_hash=digest)
        assert path.read_text(encoding="utf-8") == expected.replace("new", "")
    finally:
        await session.close()


@pytest.mark.parametrize("raw, old_text, reason", [
    (b"aaaaa", "aa", "multiple_matches"),
    (b"nothing", "absent", "no_match"),
    (b"old\xff", "old", "invalid_utf8"),
])
async def test_failed_streaming_edits_preserve_original_and_remove_staging(
    tmp_path: Path, raw: bytes, old_text: str, reason: str,
) -> None:
    path = tmp_path / "target.txt"
    path.write_bytes(raw)
    session = await LocalSandbox().open(root=tmp_path)
    try:
        with pytest.raises(SandboxOperationRejected) as error:
            await session.edit_file("target.txt", old_text, "new")
        assert error.value.safe_details["reason"] == reason
        assert path.read_bytes() == raw
        assert not list(tmp_path.glob(".linktools-????????????????????????????????"))
    finally:
        await session.close()


async def test_edit_budget_and_read_only_policy_do_not_publish_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "target.txt"
    path.write_text("old", encoding="utf-8")
    session = await LocalSandbox().open(root=tmp_path)
    read_only = await LocalSandbox(read_policy=ReadOnlySandboxPolicy(("target.txt",))).open(root=tmp_path)
    try:
        monkeypatch.setattr(_local_sandbox, "_MAX_EDIT_BYTES", 4)
        with pytest.raises(SandboxOperationRejected) as error:
            await session.edit_file("target.txt", "old", "too long")
        assert error.value.safe_details["reason"] == "edit_staging_byte_limit"
        with pytest.raises(SandboxOperationRejected):
            await read_only.edit_file("target.txt", "old", "new")
        assert path.read_text(encoding="utf-8") == "old"
    finally:
        await session.close()
        await read_only.close()


async def test_installed_libmagic_detects_bytes_despite_filename(tmp_path: Path) -> None:
    import os
    required = os.environ.get("LINKTOOLS_REQUIRE_LIBMAGIC") == "1"
    if required:
        import magic
    else:
        magic = pytest.importorskip("magic", reason="python-magic deployment is optional")
    try:
        magic.Magic(mime=True)
    except (OSError, RuntimeError) as error:
        if required:
            pytest.fail(f"Required libmagic/database unavailable: {error}")
        pytest.skip(f"System libmagic/database unavailable: {error}")
    (tmp_path / "misnamed.png").write_bytes(b"%PDF-1.7\n" + b"0" * 1024)
    session = await LocalSandbox().open(root=tmp_path)
    try:
        facts = await session.inspect_file("misnamed.png")
        assert facts["media_type"] == "application/pdf"
        assert facts["type_source"] == "conflict"
        assert not facts["is_text"]
    finally:
        await session.close()


async def test_preview_and_edit_peak_memory_do_not_scale_with_file_size(tmp_path: Path) -> None:
    import tracemalloc
    path = tmp_path / "large.txt"
    with path.open("wb") as stream:
        for _ in range(64):
            stream.write(b"x" * 65536)
        stream.write(b"unique-target")
    session = await LocalSandbox().open(root=tmp_path)
    try:
        tracemalloc.start()
        try:
            preview = await session.read_text("large.txt", max_bytes=4096)
            _, preview_peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert preview["bytes_read"] <= 65536
        assert preview_peak < 2 * 1024 * 1024
        tracemalloc.start()
        try:
            await session.edit_file("large.txt", "unique-target", "replacement")
            _, edit_peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert edit_peak < 2 * 1024 * 1024
        with path.open("rb") as stream:
            stream.seek(-11, 2)
            assert stream.read() == b"replacement"
    finally:
        await session.close()


async def test_external_change_during_edit_is_preserved_as_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "target.txt"
    path.write_text("old target", encoding="utf-8")
    check = _local_sandbox._check_state
    def mutate_before_check(target: Path, logical: str, stream: object, state: str) -> None:
        path.write_text("external update", encoding="utf-8")
        check(target, logical, stream, state)
    monkeypatch.setattr(_local_sandbox, "_check_state", mutate_before_check)
    session = await LocalSandbox().open(root=tmp_path)
    try:
        with pytest.raises(SandboxOperationRejected) as error:
            await session.edit_file("target.txt", "old", "new")
        assert error.value.code is ErrorCode.STORAGE_CONFLICT
        assert path.read_text(encoding="utf-8") == "external update"
        assert not list(tmp_path.glob(".linktools-????????????????????????????????"))
    finally:
        await session.close()


async def test_split_utf8_crlf_metadata_and_probe_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.workspace import _file_io
    monkeypatch.setattr(_file_io, "_BLOCK_BYTES", 1)
    raw = "é\r\n\n🙂\r".encode("utf-8")
    (tmp_path / "data.txt").write_bytes(raw)
    session = await LocalSandbox().open(root=tmp_path)
    try:
        result = await session.read_text("data.txt", full_metadata=True)
        assert result["content"].encode("utf-8") == raw
        assert result["total_lines"] == 3
        assert result["hash"] == hashlib.sha256(raw).hexdigest()
        facts = await session.inspect_file("data.txt", max_bytes=1)
        assert facts["bytes_read"] == 1
    finally:
        await session.close()


async def test_system_libmagic_buffer_facts_with_validation_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate the installed native detector separately from Python packaging."""
    import ctypes
    import ctypes.util
    import io
    import zipfile
    library = ctypes.util.find_library("magic")
    if library is None:
        pytest.skip("System libmagic is not installed")
    native = ctypes.CDLL(library)
    native.magic_open.argtypes = [ctypes.c_int]
    native.magic_open.restype = ctypes.c_void_p
    native.magic_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    native.magic_load.restype = ctypes.c_int
    native.magic_buffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    native.magic_buffer.restype = ctypes.c_char_p
    native.magic_close.argtypes = [ctypes.c_void_p]
    native.magic_close.restype = None
    probes: list[int] = []
    class Detector:
        def __init__(self, *, mime: bool) -> None:
            assert mime
        def from_buffer(self, sample: bytes) -> str:
            probes.append(len(sample))
            cookie = native.magic_open(0x10)
            if not cookie:
                raise RuntimeError("magic_open failed")
            try:
                if native.magic_load(cookie, None) != 0:
                    raise RuntimeError("magic database unavailable")
                buffer = ctypes.create_string_buffer(sample)
                value = native.magic_buffer(cookie, buffer, len(sample))
                if value is None:
                    raise RuntimeError("magic_buffer failed")
                return value.decode("ascii")
            finally:
                native.magic_close(cookie)
    monkeypatch.setitem(sys.modules, "magic", SimpleNamespace(Magic=Detector))
    zipped = io.BytesIO()
    with zipfile.ZipFile(zipped, "w") as archive:
        archive.writestr("payload.txt", "hello")
    fixtures = [
        ("misnamed.png", b"%PDF-1.7\n" + b"0" * 20000, "application/pdf", False),
        ("package.docx", zipped.getvalue(), ("application/zip", "application/octet-stream"), False),
        ("empty", b"", "text/plain", True),
        ("activity.log", b"ordinary business log\n", "text/plain", True),
        ("without_extension", b"ordinary text\n", "text/plain", True),
    ]
    session = await LocalSandbox().open(root=tmp_path)
    try:
        for name, raw, media_type, is_text in fixtures:
            (tmp_path / name).write_bytes(raw)
            result = await session.inspect_file(name)
            assert result["media_type"] in ((media_type,) if isinstance(media_type, str) else media_type)
            assert result["is_text"] is is_text
            assert result["bytes_read"] == min(len(raw), 8192)
        assert max(probes) == 8192
    finally:
        await session.close()


async def test_cursor_survives_session_reopen_without_rescanning_prefix(tmp_path: Path) -> None:
    text = "é🙂" * 15000 + "\r\nnext\n"
    (tmp_path / "long.txt").write_text(text, encoding="utf-8", newline="")
    session = await LocalSandbox().open(root=tmp_path)
    first = await session.read_text("long.txt", max_bytes=4096)
    await session.close()
    pieces = [first["content"]]
    current = first
    while current["next_cursor"]:
        session = await LocalSandbox().open(root=tmp_path)
        try:
            current = await session.read_text("long.txt", cursor=current["next_cursor"], max_bytes=4096)
            assert current["bytes_read"] <= 65536 + 6
            pieces.append(current["content"])
        finally:
            await session.close()
    assert "".join(pieces) == text


@pytest.mark.parametrize("lines, requested, truncated, eof", [
    (3000, 2500, True, False), (1000, 2500, False, True),
    (2000, 2500, False, True), (3000, 200, False, False),
])
async def test_line_ceiling_reports_truncation_separately_from_eof(
    tmp_path: Path, lines: int, requested: int, truncated: bool, eof: bool,
) -> None:
    (tmp_path / "lines.txt").write_text("x\n" * lines, encoding="utf-8")
    session = await LocalSandbox().open(root=tmp_path)
    try:
        result = await session.read_text("lines.txt", limit=requested)
        assert result["content"].count("\n") == min(lines, requested, 2000)
        assert result["truncated"] is truncated
        assert result["eof"] is eof
    finally:
        await session.close()


@pytest.mark.parametrize("position", [2, 999])
async def test_cursor_rejects_mid_character_and_out_of_bounds_positions(
    tmp_path: Path, position: int,
) -> None:
    from linktools.ai.workspace import _file_io
    (tmp_path / "text.txt").write_text("🙂long line", encoding="utf-8")
    session = await LocalSandbox().open(root=tmp_path)
    try:
        facts = await session.inspect_file("text.txt")
        cursor = _file_io._encode_cursor({"path": "text.txt", "state": facts["state"], "byte": position, "line": 0, "column": 1})
        with pytest.raises(AIError) as error:
            await session.read_text("text.txt", cursor=cursor)
        assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID
    finally:
        await session.close()


async def test_missing_magic_import_is_checked_once_and_keeps_extension_hints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins
    original_import = builtins.__import__
    attempts = 0
    def missing_import(name: str, *args: object, **kwargs: object) -> object:
        nonlocal attempts
        if name == "magic":
            attempts += 1
            raise ImportError("optional binding absent")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", missing_import)
    for _ in range(3):
        facts = _file_io._facts(b"header", "image.png", 6)
        assert facts["media_type"] == "image/png"
        assert facts["type_source"] == "extension"
        assert "detection_reason" not in facts
    assert attempts == 1


async def test_missing_native_library_is_disabled_once_across_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    attempts = 0
    class Detector:
        def __init__(self, *, mime: bool) -> None:
            nonlocal attempts
            attempts += 1
            raise OSError("native library or database absent")
    monkeypatch.setitem(sys.modules, "magic", SimpleNamespace(Magic=Detector))
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(lambda _: _file_io._facts(b"text", "data.txt", 4), range(9)))
    assert attempts == 1
    assert all(result["type_source"] == "extension" for result in results)
    assert all("detection_reason" not in result for result in results)


async def test_enabled_magic_is_thread_local_and_runtime_failures_remain_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor
    owners: list[int] = []
    barrier = threading.Barrier(2)
    class Detector:
        def __init__(self, *, mime: bool) -> None:
            self.owner = threading.get_ident()
            owners.append(self.owner)
        def from_buffer(self, sample: bytes) -> str:
            assert threading.get_ident() == self.owner
            if sample == b"bad":
                raise RuntimeError("one damaged sample")
            return "text/plain"
    monkeypatch.setitem(sys.modules, "magic", SimpleNamespace(Magic=Detector))
    def inspect(_: int) -> list[dict]:
        barrier.wait(timeout=5)
        return [_file_io._facts(value, "data.txt", len(value)) for value in (b"bad", b"good", b"good")]
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(inspect, range(2)))
    assert len(owners) == len(set(owners)) == 2
    for bad, first, second in results:
        assert bad["detection_reason"] == "type_detection_failed"
        assert first["type_source"] == second["type_source"] == "content"
        assert "detection_reason" not in first and "detection_reason" not in second
    assert _file_io._magic_available is True


async def test_unavailable_cache_disables_previously_initialized_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    initialized = calls = 0
    class Detector:
        def __init__(self, *, mime: bool) -> None:
            nonlocal initialized
            initialized += 1
            if initialized > 1:
                raise OSError("database unavailable for new detector")
        def from_buffer(self, sample: bytes) -> str:
            nonlocal calls
            calls += 1
            return "text/plain"
    monkeypatch.setitem(sys.modules, "magic", SimpleNamespace(Magic=Detector))
    first = _file_io._facts(b"text", "data.txt", 4)
    assert first["type_source"] == "content"
    with ThreadPoolExecutor(max_workers=1) as executor:
        second = executor.submit(_file_io._facts, b"text", "data.txt", 4).result()
    third = _file_io._facts(b"text", "data.txt", 4)
    assert second["type_source"] == third["type_source"] == "extension"
    assert initialized == 2 and calls == 1


@pytest.mark.parametrize("detected, name, expected_type, expected_source", [
    ("text/plain", "data.txt", "text/plain", "content"),
    ("text/plain", "data.json", "application/json", "extension"),
    ("application/xml", "data.json", "application/xml", "conflict"),
])
async def test_generic_text_hints_do_not_hide_structured_content_conflicts(
    monkeypatch: pytest.MonkeyPatch, detected: str, name: str,
    expected_type: str, expected_source: str,
) -> None:
    class Detector:
        def __init__(self, *, mime: bool) -> None:
            assert mime
        def from_buffer(self, sample: bytes) -> str:
            return detected
    monkeypatch.setitem(sys.modules, "magic", SimpleNamespace(Magic=Detector))
    result = _file_io._facts(b"text", name, 4)
    assert result["is_text"]
    assert result["media_type"] == expected_type
    assert result["type_source"] == expected_source
