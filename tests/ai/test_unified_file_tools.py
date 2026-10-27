#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified model-facing file operations and publication boundaries."""
from linktools.ai.model import create_model

from ._runtime_test_helpers import workspace_features

import base64
import hashlib
import json
from pathlib import Path


import pytest
from pydantic_ai.messages import BinaryContent, ToolReturn

from linktools.ai.capability import ToolCallRetry, filesystem_capabilities
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import PromptLimits
from linktools.ai.errors import AIError, ErrorCode

from linktools.ai.workspace import LocalWorkspaceBackend, WorkspaceReadPolicy, Workspace

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6"
    "SgAAAABJRU5ErkJggg=="
)


def _binding():
    return create_model(api='openai',
        model="explicit-test",
        input_types=(
            "mime:image/png",
            "mime:text/plain",
            "mime:text/csv",
        ),
    )


def _tools(root: Path, session, *, limits=None, binding=None, names=None):
    capability = filesystem_capabilities(
        Workspace.load(root), names or ("read_file", "write_file", "list_files", "search_files"),
        backend=session, limits=limits, model_binding=binding or _binding(),
    )[0]
    return capability.get_toolset().tools


@pytest.mark.asyncio
async def test_unified_read_mixed_batch_preserves_input_order_and_duplicate_occurrences(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(_PNG)
    session = LocalWorkspaceBackend(tmp_path)
    await session.working_dir()
    try:
        tools = _tools(tmp_path, session)
        result = await tools["read_file"].function(["image.png", "notes.txt", "image.png"])
        assert isinstance(result, ToolReturn)
        assert [item["input_index"] for item in result.return_value[0]["files"]] == [0, 2]
        assert [item["input_index"] for item in result.return_value[0]["texts"]] == [1]
        assert "alpha" in result.return_value[0]["texts"][0]["content"]
        assert result.return_value[0]["texts"][0]["mime_type"] == "text/plain"
        assert "media_type" not in result.return_value[0]["texts"][0]
        assert all(item["mime_type"] == "image/png" and "media_type" not in item
                   for item in result.return_value[0]["files"])
        assert "hash" not in result.return_value[0]["texts"][0]
        assert result.content is None
        assert len(result.return_value[1:]) == 2
        assert all(isinstance(item, BinaryContent) and item.data == _PNG for item in result.return_value[1:])
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_unified_read_singleton_array_and_cursor_preserve_text(tmp_path: Path) -> None:
    body = "\n".join(f"line-{n}" for n in range(250))
    (tmp_path / "notes.txt").write_text(body, encoding="utf-8")
    session = LocalWorkspaceBackend(tmp_path)
    await session.working_dir()
    try:
        read = _tools(tmp_path, session)["read_file"].function
        first = (await read(["notes.txt"], limit=2)).return_value["texts"][0]
        second = (await read("notes.txt", cursor=first["next_cursor"], limit=2)).return_value["texts"][0]
        assert first["eof"] is False
        assert first["truncated"] is False
        assert first["offset"] == 0
        assert second["offset"] == 2
        assert "line-2" in second["content"] and "line-0" not in second["content"]
        full = (await read("notes.txt", full_metadata=True)).return_value["texts"][0]
        assert full["hash"] == hashlib.sha256(body.encode()).hexdigest()
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_unified_read_explicit_text_attachment_preserves_mime_type(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("whole text", encoding="utf-8")
    session = LocalWorkspaceBackend(tmp_path)
    await session.working_dir()
    try:
        result = await _tools(tmp_path, session)["read_file"].function("notes.txt", as_file=True)
        assert result.return_value[0]["texts"] == []
        assert result.return_value[0]["files"][0]["mime_type"] == "text/plain"
        assert result.content is None
        assert result.return_value[1].data == b"whole text"
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("paths,kwargs", [
    (["notes.txt", "notes.txt"], {"limit": 2}),
    ("notes.txt", {"as_file": True, "offset": 0}),
    ("image.png", {"limit": 1}),
    ("image.png", {"full_metadata": True}),
    ("notes.txt", {"cursor": "opaque", "offset": 0}),
    ([], {}),
])
async def test_unified_read_rejects_invalid_conditional_arguments(tmp_path: Path, paths, kwargs) -> None:
    (tmp_path / "notes.txt").write_text("text", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(_PNG)
    session = LocalWorkspaceBackend(tmp_path)
    await session.working_dir()
    try:
        with pytest.raises(ToolCallRetry):
            await _tools(tmp_path, session)["read_file"].function(paths, **kwargs)
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_unified_read_later_failure_returns_no_partial_batch(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("private preview", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(_PNG)
    session = LocalWorkspaceBackend(tmp_path)
    await session.working_dir()
    try:
        denied = create_model(api='openai', model="no-inputs", input_types=())
        with pytest.raises(ToolCallRetry) as rejected:
            await _tools(tmp_path, session, binding=denied)["read_file"].function(["notes.txt", "image.png"])
        facts = json.loads(str(rejected.value).split(" File: ", 1)[1])
        assert facts == {"reason": "model_input_not_supported", "mime_type": "image/png",
                         "size": len(_PNG), "path": "image.png"}
        assert "private preview" not in str(rejected.value)
        with pytest.raises(ToolCallRetry):
            await _tools(tmp_path, session)["read_file"].function(["notes.txt", "missing.txt"])
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_unified_read_counts_texts_against_batch_limits(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("text", encoding="utf-8")
    session = LocalWorkspaceBackend(tmp_path)
    await session.working_dir()
    try:
        tools = _tools(tmp_path, session, limits=PromptLimits(max_binary_input_parts=1))
        with pytest.raises(ToolCallRetry):
            await tools["read_file"].function(["notes.txt", "notes.txt"])
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_unified_write_optional_hash_and_exact_replacement(tmp_path: Path) -> None:
    session = LocalWorkspaceBackend(tmp_path)
    await session.working_dir()
    try:
        write = _tools(tmp_path, session)["write_file"].function
        await write("notes.txt", "before target after")
        await write("notes.txt", "replacement", old_text="target")
        assert (tmp_path / "notes.txt").read_text() == "before replacement after"
        expected = hashlib.sha256((tmp_path / "notes.txt").read_bytes()).hexdigest()
        await write("notes.txt", "", old_text="replacement", expected_hash=expected)
        assert (tmp_path / "notes.txt").read_text() == "before  after"
        with pytest.raises(ToolCallRetry):
            await write("notes.txt", "bad", old_text="")
        with pytest.raises(ToolCallRetry):
            await write("notes.txt", "bad", expected_hash=expected)
        with pytest.raises(ToolCallRetry):
            await write("missing.txt", "bad", old_text="target")
        await write("notes.txt", "")
        assert (tmp_path / "notes.txt").read_bytes() == b""
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_unified_file_selection_keeps_scenarios_minimal(tmp_path: Path) -> None:
    session = LocalWorkspaceBackend(tmp_path)
    await session.working_dir()
    try:
        for names in (("read_file", "search_files"), ("read_file", "search_files", "list_files"),
                      ("read_file", "search_files", "list_files", "write_file")):
            assert set(_tools(tmp_path, session, names=names)) == set(names)
        narrow = _tools(tmp_path, session, names=("edit_file",))
        assert set(narrow) == {"edit_file"}
        assert "old_text" in narrow["edit_file"].tool_def.parameters_json_schema["required"]
    finally:
        await session.close()


def test_workspace_tools_use_initial_behavior_revision(tmp_path: Path) -> None:
    contributions = workspace_features("workspace", workspace=Workspace.load(tmp_path))._contributions
    revisions = {item.id: item.revision for item in contributions}
    assert {name: revisions[name] for name in ("read_file", "write_file", "search_files")} == {
        "read_file": 1, "write_file": 1, "search_files": 1,
    }
    assert revisions["attach_files"] == 1
    assert revisions["edit_file"] == revisions["list_files"] == 1


@pytest.mark.asyncio
async def test_legacy_attach_obeys_explicit_model_route_before_publication(tmp_path: Path) -> None:
    (tmp_path / "report.pdf").write_bytes(b"%PDF-1.7\n")
    binding = create_model(api='openai', model="no-files", input_types=())
    session = LocalWorkspaceBackend(tmp_path)
    await session.working_dir()
    try:
        with pytest.raises(ToolCallRetry):
            await _tools(tmp_path, session, binding=binding, names=("attach_files",))["attach_files"].function(["report.pdf"])
    finally:
        await session.close()


def test_parallel_read_results_share_request_budget_without_counting_old_previews() -> None:
    from pydantic_ai.messages import ModelRequest, ModelResponse, ToolReturnPart, TextPart
    from linktools.ai.runtime._compaction import _validate_pending_binary_content
    result = {"files": [], "texts": [{"path": "notes.txt", "content": "x" * 100}]}
    old = ModelRequest(parts=[ToolReturnPart("read_file", result, "old")])
    response = ModelResponse(parts=[TextPart("next")])
    one = ModelRequest(parts=[ToolReturnPart("read_file", result, "a")])
    two = ModelRequest(parts=[ToolReturnPart("read_file", result, "b")])
    limits = PromptLimits(max_binary_input_parts=1, max_file_output_bytes=250)
    _validate_pending_binary_content([old, response, one], limits)
    with pytest.raises(AIError) as error:
        _validate_pending_binary_content([response, one, two], limits)
    assert error.value.safe_details["reason"] == "file_request_budget_exceeded"
    with pytest.raises(AIError):
        _validate_pending_binary_content([response, one, two], PromptLimits(max_file_output_bytes=250))
    interrupted = ModelResponse(parts=[TextPart("partial")], state="incomplete")
    with pytest.raises(AIError):
        _validate_pending_binary_content([response, one, two, interrupted], limits)


@pytest.mark.asyncio
async def test_explicit_skill_cli_converts_new_format_without_core_parser(tmp_path: Path) -> None:
    import json
    import shlex
    import sys
    from linktools.ai.capability import SkillCapability, SkillDefinition, SkillSourceRegistry
    from linktools.ai.spec import SkillSpec
    example = Path(__file__).parents[2] / "examples" / "skills" / "email-preview"
    skill = SkillCapability(
        (SkillDefinition(SkillSpec("email-preview", content=(example / "SKILL.md").read_text())),),
        SkillSourceRegistry(),
    )
    assert set(skill.get_toolset().tools) == {"list_skills", "load_skill"}
    loaded = await skill.load_skill("email-preview")
    assert "authorized" in str(loaded)
    (tmp_path / "extract_email.py").write_bytes((example / "extract_email.py").read_bytes())
    (tmp_path / "message.eml").write_text("From: sender@example.test\nSubject: Example\nContent-Type: text/plain; charset=utf-8\n\nEvidence body", encoding="utf-8")
    session = LocalWorkspaceBackend(tmp_path)
    await session.working_dir()
    try:
        with pytest.raises(ToolCallRetry):
            await _tools(tmp_path, session)["read_file"].function("message.eml")
        result = await session.run_command(f"{shlex.quote(sys.executable)} extract_email.py message.eml preview.txt", timeout_seconds=5)
        assert (tmp_path / "preview.txt").exists(), result
        preview = await _tools(tmp_path, session)["read_file"].function("preview.txt")
        assert preview.return_value["texts"][0]["content"] == "Evidence body"
        provenance = json.loads((tmp_path / "preview.txt.source.json").read_text())
        assert provenance["source"] == "message.eml" and provenance["lossy"] is True
        assert "attachments" in provenance["omitted"]
        (tmp_path / "bad.eml").write_text("not email", encoding="utf-8")
        await session.run_command(f"{shlex.quote(sys.executable)} extract_email.py bad.eml bad.txt", timeout_seconds=5)
        assert not (tmp_path / "bad.txt").exists()
    finally:
        await session.close()
    restricted = LocalWorkspaceBackend(tmp_path, read_policy=WorkspaceReadPolicy(("*.eml",)))
    await restricted.working_dir()
    try:
        with pytest.raises(AIError) as rejected:
            await restricted.run_command("python extract_email.py message.eml blocked.txt", timeout_seconds=5)
        assert rejected.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert not (tmp_path / "blocked.txt").exists()
    finally:
        await restricted.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("reason,expected", [
    ("model_input_not_supported", "explicitly does not support"),
    ("input_encoding_not_supported", "endpoint cannot encode"),
    ("text_input_not_utf8", "valid UTF-8"),
    ("file_batch_timeout", "time budget"),
    ("file_batch_invalid", "nonempty path array"),
    ("stale_cursor", "source changed"),
])
async def test_read_errors_explain_the_actual_recovery_step(tmp_path: Path, reason, expected) -> None:
    from types import SimpleNamespace
    async def reject(*args, **kwargs):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"reason": reason})
    from ._runtime_test_helpers import FileToolsStub
    backend = FileToolsStub()
    backend.inspect_file = reject
    tools = _tools(tmp_path, backend)
    with pytest.raises(ToolCallRetry, match=expected):
        await tools["read_file"].function("file.txt")


@pytest.mark.asyncio
@pytest.mark.parametrize("name,mime", [("data.csv", "text/csv"), ("data.yaml", "application/yaml")])
async def test_generic_text_detection_keeps_structured_text_attachment_hint(tmp_path: Path, monkeypatch, name, mime) -> None:
    import sys
    from types import SimpleNamespace
    class Detector:
        def __init__(self, **kwargs): pass
        def from_buffer(self, sample): return "text/plain"
    from linktools.ai.workspace import _file_io
    monkeypatch.setattr(_file_io, "_magic_available", None)
    monkeypatch.setattr(_file_io, "_magic_factory", None)
    monkeypatch.setattr(_file_io, "_magic_threads", _file_io._MagicThreadState())
    monkeypatch.setitem(sys.modules, "magic", SimpleNamespace(Magic=Detector))
    # MIME database variations should not change the subtype-refinement invariant.
    import mimetypes
    mimetypes.add_type(mime, Path(name).suffix)
    (tmp_path / name).write_text("key,value\none,two\n", encoding="utf-8")
    binding = create_model(api='openai', model="known-text", input_types=(f"mime:{mime}",))
    session = LocalWorkspaceBackend(tmp_path)
    await session.working_dir()
    try:
        value = (await _tools(tmp_path, session, binding=binding)["read_file"].function(name, as_file=True)).return_value[0]
        assert value["files"][0]["mime_type"] == mime
        assert "media_type" not in value["files"][0]
        assert value["files"][0]["type_source"] == "extension"
        assert value["files"][0]["mime_type"] == mime
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("name,body,mime_type,as_file", [
    ("report.pdf", b"%PDF-1.7\n", "application/pdf", False),
    ("report.docx", b"PK\x03\x04\x00", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", False),
    ("notes.txt", b"private whole text", "text/plain", True),
])
async def test_unsupported_attachment_reports_facts_without_content_or_native_detector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, body: bytes,
    mime_type: str, as_file: bool,
) -> None:
    from linktools.ai.workspace import _file_io
    monkeypatch.setattr(_file_io, "_magic_available", False)
    (tmp_path / name).write_bytes(body)
    session = LocalWorkspaceBackend(tmp_path)
    await session.working_dir()
    try:
        denied = create_model(api='openai', model="no-inputs")
        with pytest.raises(ToolCallRetry) as rejected:
            await _tools(tmp_path, session, binding=denied)["read_file"].function(name, as_file=as_file)
        facts = json.loads(str(rejected.value).split(" File: ", 1)[1])
        assert facts == {"reason": "model_input_not_supported", "mime_type": mime_type,
                         "size": len(body), "path": name}
        assert str(tmp_path) not in str(rejected.value)
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_default_attachment_deny_keeps_bounded_text_preview_available(tmp_path: Path) -> None:
    (tmp_path / "large.txt").write_text("line\n" * 10000, encoding="utf-8")
    session = LocalWorkspaceBackend(tmp_path)
    await session.working_dir()
    try:
        denied = create_model(api='openai', model="no-inputs")
        result = await _tools(tmp_path, session, binding=denied)["read_file"].function("large.txt", offset=1, limit=2)
        assert result.return_value["texts"][0]["content"] == "line\nline\n"
        assert result.return_value["files"] == []
        assert not result.content
    finally:
        await session.close()
