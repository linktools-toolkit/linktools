#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stream-attempt display preserves realtime output without merging failed answers."""

import asyncio
import io
import json
import re
import shutil
import subprocess
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from collections.abc import Sequence

import pytest
from rich.console import Console

from linktools.cli import CommandError
from linktools.commands.ai import run


def event(kind: str, text: str | None = None, *, attempt: int | None = 1,
          request: int = 1, agent_run: int = 1, depth: int = 0,
          execution: str = "execution", **extra: object) -> SimpleNamespace:
    payload: dict[str, object] = dict(extra)
    if attempt is not None:
        payload.update(agent_run_seq=agent_run, model_request_seq=request, stream_attempt_seq=attempt)
    if text is not None:
        payload["text"] = text
    return SimpleNamespace(depth=depth, execution_id=execution,
                           event=SimpleNamespace(event_type=kind, payload=payload))


def output(monkeypatch: pytest.MonkeyPatch, terminal: bool = False) -> tuple[io.StringIO, io.StringIO]:
    stdout, stderr = io.StringIO(), io.StringIO()
    console = Console(file=stdout, force_terminal=terminal, width=2000)
    diagnostics = Console(file=stderr, force_terminal=terminal, width=2000)
    monkeypatch.setattr(run, "get_console", lambda: console)
    monkeypatch.setattr(run, "Console", lambda **_: diagnostics)
    return stdout, stderr


def plain(value: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", value)


def model_item(text: str, request: int = 1) -> SimpleNamespace:
    return SimpleNamespace(execution_id="execution", agent_run_seq=1, model_request_seq=request,
                           status="SUCCEEDED", content_included=True,
                           response={"parts": [{"part_kind": "text", "content": text}]})


class Execution:
    execution_id = "execution"

    def __init__(self, events: list[SimpleNamespace], models: Sequence[object] = ()) -> None:
        self.events = events
        self.models = models

    async def model_interactions(self, *, include_content: bool, agent_run_seq: int, model_request_seq: int, limit: int) -> SimpleNamespace:
        assert include_content and limit == 1
        return SimpleNamespace(items=tuple(item for item in self.models if item.agent_run_seq == agent_run_seq and item.model_request_seq == model_request_seq))

    async def watch(self, *, include_content: bool):
        assert include_content
        for item in self.events:
            yield item


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("attempt", [None, 1])
async def test_cli_normal_stream_output_is_unchanged(monkeypatch: pytest.MonkeyPatch, terminal: bool, attempt: int | None) -> None:
    stdout, stderr = output(monkeypatch, terminal)
    assert await run._stream_result(Execution([
        event("MODEL_ATTEMPT_STARTED", attempt=attempt),
        event("ASSISTANT_TEXT_DELTA", "hello ", attempt=attempt),
        event("ASSISTANT_THINKING_DELTA", "thought", attempt=attempt),
        event("ASSISTANT_TEXT_DELTA", "world", attempt=attempt),
        event("MODEL_ATTEMPT_FINISHED", attempt=attempt, status="SUCCEEDED"),
        event("MODEL_REQUEST_FINISHED", attempt=attempt, status="SUCCEEDED"),
        event("EXECUTION_SUCCEEDED"),
    ])) == 0
    assert plain(stdout.getvalue()) == "hello world\n"
    assert "[thinking] thought" in plain(stderr.getvalue())


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", [False, True])
async def test_cli_retries_invalidate_partial_on_stdout_before_replacement(monkeypatch: pytest.MonkeyPatch, terminal: bool) -> None:
    stdout, _ = output(monkeypatch, terminal)
    first_visible, release = asyncio.Event(), asyncio.Event()

    class GatedExecution:
        execution_id = "execution"

        async def model_interactions(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(items=[model_item("winner")])

        async def watch(self, *, include_content: bool):
            assert include_content
            yield event("MODEL_ATTEMPT_STARTED")
            yield event("ASSISTANT_TEXT_DELTA", "old-partial")
            first_visible.set()
            await release.wait()
            yield event("MODEL_ATTEMPT_FINISHED", status="FAILED", retry_scheduled=True)
            yield event("MODEL_ATTEMPT_STARTED", attempt=2, replaces_stream_attempt_seq=1)
            yield event("ASSISTANT_TEXT_DELTA", "second-partial", attempt=2)
            yield event("MODEL_ATTEMPT_FINISHED", attempt=2, status="FAILED", retry_scheduled=True)
            yield event("MODEL_ATTEMPT_STARTED", attempt=3, replaces_stream_attempt_seq=2)
            yield event("ASSISTANT_TEXT_DELTA", "winner", attempt=3)
            yield event("MODEL_REQUEST_FINISHED", attempt=3, status="SUCCEEDED")
            yield event("EXECUTION_SUCCEEDED")

    task = asyncio.create_task(run._stream_result(GatedExecution()))
    try:
        await asyncio.wait_for(first_visible.wait(), 2)
        assert stdout.getvalue() == "old-partial"
        assert not task.done()
        release.set()
        assert await asyncio.wait_for(task, 2) == 0
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
    value = plain(stdout.getvalue())
    assert value.count("INVALIDATED") == 2
    assert value.count("winner") == 2
    assert value.index("old-partial") < value.index("attempt 1 FAILED") < value.index("attempt 2 begins") < value.index("second-partial")
    assert value.index("attempt 2 FAILED") < value.index("attempt 3 begins") < value.index("winner") < value.index("request 1 COMPLETED")
    assert "old-partialsecond-partialwinner" not in value
    if not terminal:
        assert "\x1b" not in value


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["FAILED", "CANCELLED"])
async def test_cli_terminal_attempt_is_not_a_completed_answer(monkeypatch: pytest.MonkeyPatch, status: str) -> None:
    stdout, _ = output(monkeypatch)
    with pytest.raises(CommandError):
        await run._stream_result(Execution([
            event("ASSISTANT_TEXT_DELTA", "uncommitted"),
            event("MODEL_ATTEMPT_FINISHED", status=status, retry_scheduled=False),
            event("MODEL_REQUEST_FINISHED", status=status),
            event("EXECUTION_" + status),
        ]))
    assert plain(stdout.getvalue()).count("INVALIDATED") == 1
    assert status in stdout.getvalue()
    assert "COMPLETED" not in stdout.getvalue()


@pytest.mark.asyncio
async def test_cli_ignores_old_attempts_and_does_not_clear_other_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout, _ = output(monkeypatch)
    assert await run._stream_result(Execution([
        event("ASSISTANT_TEXT_DELTA", "earlier"),
        event("MODEL_REQUEST_FINISHED", status="SUCCEEDED"),
        event("ASSISTANT_TEXT_DELTA", "discard", request=2),
        event("MODEL_ATTEMPT_STARTED", attempt=2, request=2),
        event("MODEL_ATTEMPT_STARTED", request=2),
        event("ASSISTANT_TEXT_DELTA", "stale", request=2),
        event("MODEL_ATTEMPT_FINISHED", request=2, status="FAILED"),
        event("MODEL_REQUEST_FINISHED", request=2, status="SUCCEEDED"),
        event("ASSISTANT_TEXT_DELTA", "child", depth=1, execution="child"),
        event("ASSISTANT_TEXT_DELTA", "replacement", attempt=2, request=2),
        event("MODEL_REQUEST_FINISHED", attempt=2, request=2, status="SUCCEEDED"),
        event("MODEL_ATTEMPT_FINISHED", attempt=2, request=2, status="FAILED"),
        event("ASSISTANT_TEXT_DELTA", "new-run", agent_run=2),
        event("MODEL_REQUEST_FINISHED", agent_run=2, status="SUCCEEDED"),
        event("EXECUTION_SUCCEEDED"),
    ])) == 0
    value = stdout.getvalue()
    assert value.startswith("earlierdiscard\n")
    assert "stale" not in value and "child" not in value
    assert "request 1 attempt 1 FAILED" not in value
    assert value.count("INVALIDATED") == 1
    assert "COMPLETED" not in value and "FINAL UNAVAILABLE" in value
    assert "replacement" in value and "new-run\n" in value


@pytest.mark.asyncio
async def test_cli_failed_attempt_without_text_adds_no_stdout_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout, stderr = output(monkeypatch)
    assert await run._stream_result(Execution([
        event("MODEL_ATTEMPT_STARTED"),
        event("ASSISTANT_THINKING_DELTA", "partial thought"),
        event("MODEL_ATTEMPT_FINISHED", status="FAILED", retry_scheduled=True),
        event("MODEL_ATTEMPT_STARTED", attempt=2),
        event("ASSISTANT_TEXT_DELTA", "answer", attempt=2),
        event("MODEL_REQUEST_FINISHED", attempt=2, status="SUCCEEDED"),
        event("EXECUTION_SUCCEEDED"),
    ])) == 0
    assert stdout.getvalue() == "answer\n"
    assert "thinking invalidated" in stderr.getvalue()


@pytest.mark.asyncio
async def test_cli_preview_gap_is_not_an_attempt_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout, _ = output(monkeypatch)
    assert await run._stream_result(Execution([
        event("ASSISTANT_TEXT_DELTA", "prefix"),
        event("STREAM_RESET", attempt=None),
        event("MODEL_ATTEMPT_STARTED"),
        event("ASSISTANT_TEXT_DELTA", "tail", stream_truncated=True),
        event("MODEL_REQUEST_FINISHED", status="SUCCEEDED"),
        event("EXECUTION_SUCCEEDED"),
    ])) == 0
    value = stdout.getvalue()
    assert "prefixtail" not in value
    assert value.index("prefix") < value.index("preview GAP") < value.index("tail")
    assert "no retry implied" in value and "preview is incomplete" in value
    assert "INVALIDATED" not in value and "attempt 2" not in value


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("started", [False, True])
async def test_cli_gap_between_transports_separates_successful_prefix(monkeypatch: pytest.MonkeyPatch, terminal: bool, started: bool) -> None:
    stdout, _ = output(monkeypatch, terminal)
    items = [event("ASSISTANT_TEXT_DELTA", "completed other request", request=2),
             event("MODEL_REQUEST_FINISHED", request=2, status="SUCCEEDED"),
             event("ASSISTANT_TEXT_DELTA", "old snapshot", part_index=0, part_start=True),
             event("MODEL_ATTEMPT_FINISHED", status="SUCCEEDED")]
    if started:
        items.append(event("MODEL_ATTEMPT_STARTED", attempt=2, replaces_stream_attempt_seq=None))
    items += [event("STREAM_RESET", attempt=None),
              event("ASSISTANT_TEXT_DELTA", "retained tail", attempt=2, part_index=0, part_start=False),
              event("MODEL_ATTEMPT_FINISHED", attempt=2, status="SUCCEEDED"),
              event("MODEL_REQUEST_FINISHED", attempt=2, status="SUCCEEDED"),
              event("EXECUTION_SUCCEEDED")]
    assert await run._stream_result(Execution(items, [model_item("canonical after gap")])) == 0
    value = plain(stdout.getvalue())
    assert value.index("old snapshot") < value.index("request 1 preview GAP") < value.index("retained tail")
    assert value.count("preview GAP") == 1
    assert "request 2 preview GAP" not in value
    assert "preview is incomplete" in value
    assert "UPDATED" not in value and "INVALIDATED" not in value and "FAILED" not in value


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("retry_initial", [False, True])
async def test_cli_successful_continuation_preserves_previous_segments(monkeypatch: pytest.MonkeyPatch, terminal: bool, retry_initial: bool) -> None:
    stdout, _ = output(monkeypatch, terminal)
    seq = 2 if retry_initial else 1
    items = [event("MODEL_ATTEMPT_STARTED")]
    if retry_initial:
        items += [event("ASSISTANT_TEXT_DELTA", "discard"),
                  event("MODEL_ATTEMPT_FINISHED", status="FAILED", retry_scheduled=True),
                  event("MODEL_ATTEMPT_STARTED", attempt=seq, replaces_stream_attempt_seq=1)]
    items += [event("ASSISTANT_TEXT_DELTA", "prefix ", attempt=seq, part_index=0, part_start=True),
              event("MODEL_ATTEMPT_FINISHED", attempt=seq, status="SUCCEEDED"),
              event("MODEL_ATTEMPT_STARTED", attempt=seq + 1, replaces_stream_attempt_seq=None),
              event("ASSISTANT_TEXT_DELTA", "suffix", attempt=seq + 1, part_index=1, part_start=True),
              event("MODEL_ATTEMPT_FINISHED", attempt=seq + 1, status="SUCCEEDED"),
              event("MODEL_REQUEST_FINISHED", attempt=seq + 1, status="SUCCEEDED"),
              event("EXECUTION_SUCCEEDED")]
    assert await run._stream_result(Execution(items, [model_item("prefix suffix")])) == 0
    value = plain(stdout.getvalue())
    assert "prefix suffix" in value
    assert f"attempt {seq + 1} begins" not in value
    assert f"attempt {seq} FAILED" not in value
    if retry_initial:
        assert value.count("INVALIDATED") == 1
        assert value.count("COMPLETED") == 1
    else:
        assert value == "prefix suffix\n"


@pytest.mark.asyncio
async def test_cli_failed_continuation_does_not_invalidate_successful_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout, _ = output(monkeypatch)
    with pytest.raises(CommandError):
        await run._stream_result(Execution([
            event("ASSISTANT_TEXT_DELTA", "completed prefix "),
            event("MODEL_ATTEMPT_FINISHED", status="SUCCEEDED"),
            event("MODEL_ATTEMPT_STARTED", attempt=2, replaces_stream_attempt_seq=None),
            event("ASSISTANT_TEXT_DELTA", "partial suffix", attempt=2),
            event("MODEL_ATTEMPT_FINISHED", attempt=2, status="FAILED", retry_scheduled=False),
            event("MODEL_REQUEST_FINISHED", attempt=2, status="FAILED"),
            event("EXECUTION_FAILED"),
        ]))
    value = stdout.getvalue()
    assert value.startswith("completed prefix partial suffix\n")
    assert "attempt 1 FAILED" not in value
    assert "attempt 2 FAILED" in value and "14 streamed text characters INVALIDATED" in value
    assert value.count("logical response incomplete") == 1
    assert "COMPLETED" not in value


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["updated ", ""])
async def test_cli_part_update_preserves_other_successful_parts(monkeypatch: pytest.MonkeyPatch, replacement: str) -> None:
    stdout, _ = output(monkeypatch)
    items = [event("ASSISTANT_TEXT_DELTA", "old ", part_index=0, part_start=True),
             event("ASSISTANT_TEXT_DELTA", "keep", part_index=1, part_start=True),
             event("MODEL_ATTEMPT_FINISHED", status="SUCCEEDED"),
             event("MODEL_ATTEMPT_STARTED", attempt=2, replaces_stream_attempt_seq=None),
             event("ASSISTANT_TEXT_DELTA", replacement, attempt=2, part_index=0, part_start=True),
             event("MODEL_ATTEMPT_FINISHED", attempt=2, status="SUCCEEDED"),
             event("MODEL_REQUEST_FINISHED", attempt=2, status="SUCCEEDED"),
             event("EXECUTION_SUCCEEDED")]
    for item in items:
        item.event.payload = MappingProxyType(item.event.payload)
    assert await run._stream_result(Execution(items)) == 0
    value = stdout.getvalue()
    assert value.startswith("old keep\n")
    assert "part 0 UPDATED" in value and "part 1 UPDATED" not in value
    assert "INVALIDATED" not in value and "FAILED" not in value
    assert value.count("keep") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize(("readback", "head"), [("available", "old-head "), ("missing", "old-head "), ("error", "old-head "), ("other-request", "old-head "), ("available", "")])
async def test_cli_final_readback_supersedes_all_earlier_preview_parts(monkeypatch: pytest.MonkeyPatch, terminal: bool, readback: str, head: str) -> None:
    stdout, stderr = output(monkeypatch, terminal)
    items = [event("ASSISTANT_TEXT_DELTA", head, part_index=0, part_start=True),
             event("ASSISTANT_TEXT_DELTA", "old-tail", part_index=1, part_start=True),
             event("MODEL_ATTEMPT_FINISHED", status="SUCCEEDED"),
             event("MODEL_ATTEMPT_STARTED", attempt=2, replaces_stream_attempt_seq=None),
             event("ASSISTANT_TEXT_DELTA", "new-only", attempt=2, part_index=0, part_start=True),
             event("MODEL_ATTEMPT_FINISHED", attempt=2, status="SUCCEEDED"),
             event("MODEL_REQUEST_FINISHED", attempt=2, status="SUCCEEDED"),
             event("EXECUTION_SUCCEEDED")]

    class ReadbackExecution(Execution):
        async def model_interactions(self, **kwargs: object) -> SimpleNamespace:
            assert kwargs == {"include_content": True, "agent_run_seq": 1, "model_request_seq": 1, "limit": 1}
            if readback == "error":
                raise RuntimeError("synthetic reader failure")
            return SimpleNamespace(items=[model_item("new-only", 2 if readback == "other-request" else 1)] if readback != "missing" else [])

    assert await run._stream_result(ReadbackExecution(items)) == 0
    value = plain(stdout.getvalue())
    assert value.startswith(head + "old-tail\n")
    if head:
        assert "part 0 UPDATED" in value
    assert "SUCCEEDED" in plain(stderr.getvalue()), "readback failure cannot change the execution verdict"
    if readback == "available":
        final = value.split("CANONICAL TEXT RESPONSE:", 1)[1]
        assert "all earlier preview text for this request is superseded" in final
        assert "\nnew-only\n" in final and "old-tail" not in final and "old-head" not in final
        assert "COMPLETED: committed SDK text above" in final
    else:
        assert "FINAL UNAVAILABLE" in value and "COMPLETED" not in value


@pytest.mark.asyncio
@pytest.mark.parametrize("same_id", [False, True])
async def test_real_runtime_continuation_events_reach_cli_and_web(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, same_id: bool) -> None:
    from .test_model_stream_retry import runtime_continuation_retry_events

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the Web projection replay")
    observed, models = await runtime_continuation_retry_events(same_id, include_models=True)
    stdout, _ = output(monkeypatch)
    items = [SimpleNamespace(depth=0, execution_id=item.execution_id, event=item) for item in observed]
    execution = Execution(items, models)
    execution.execution_id = items[0].execution_id
    assert await run._stream_result(execution) == 0
    value = stdout.getvalue()
    assert value.count("INVALIDATED") == 1
    assert "attempt 2 FAILED" not in value
    canonical = value.split("CANONICAL TEXT RESPONSE:", 1)[1]
    expected = "".join(part["content"] for part in models[0].response["parts"] if part["part_kind"] == "text")
    assert f"\n{expected}\n" in canonical and "discarded" not in canonical
    if same_id:
        assert value.index("segment-one") < value.index("part 0 UPDATED") < value.index("segment-two")
        assert "segment-one segment-two" not in value
    else:
        assert "segment-one segment-two" in value and "UPDATED" not in value
    wire = [{"execution_id":item.execution_id,"event":{"event_type":item.event_type,"payload":item.payload}}
            for item in observed]
    fixture = tmp_path / "stream-events.json"
    fixture.write_text(json.dumps(wire, default=dict))
    root = Path(__file__).parents[2]
    result = subprocess.run([node, str(Path(__file__).with_name("web_console_attempts.mjs")),
                             str(root / "linktools-ai/src/linktools/ai/web/assets/console.js"), str(fixture),
                             "segment-two" if same_id else "segment-one segment-two"],
                            capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_real_runtime_empty_start_reaches_cli_and_web(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from .test_model_stream_retry import runtime_empty_start_events

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the Web projection replay")
    observed, models = await runtime_empty_start_events(include_models=True)
    deltas = [item.payload for item in observed if item.event_type == "ASSISTANT_TEXT_DELTA"]
    assert [(item["text"], item["part_start"]) for item in deltas] == [("", True), ("winner", False)]
    stdout, _ = output(monkeypatch)
    items = [SimpleNamespace(depth=0, execution_id=item.execution_id, event=item) for item in observed]
    execution = Execution(items, models)
    execution.execution_id = items[0].execution_id
    assert await run._stream_result(execution) == 0
    assert stdout.getvalue() == "winner\n"
    fixture = tmp_path / "empty-start-events.json"
    wire = [{"execution_id":item.execution_id,"event":{"event_type":item.event_type,"payload":item.payload}}
            for item in observed]
    fixture.write_text(json.dumps(wire, default=dict))
    root = Path(__file__).parents[2]
    result = subprocess.run([node, str(Path(__file__).with_name("web_console_attempts.mjs")),
                             str(root / "linktools-ai/src/linktools/ai/web/assets/console.js"), str(fixture),
                             "winner", "empty-start"], capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
@pytest.mark.parametrize("closed_segment", [False, True])
async def test_cli_interrupt_marks_visible_attempt_unconfirmed(monkeypatch: pytest.MonkeyPatch, closed_segment: bool) -> None:
    stdout, _ = output(monkeypatch)
    waiting = asyncio.Event()

    class PendingExecution:
        execution_id = "execution"

        async def watch(self, *, include_content: bool):
            assert include_content
            yield event("ASSISTANT_TEXT_DELTA", "partial")
            if closed_segment:
                yield event("MODEL_ATTEMPT_FINISHED", status="SUCCEEDED")
            waiting.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(run._stream_result(PendingExecution()))
    await asyncio.wait_for(waiting.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert "INTERRUPTED" in stdout.getvalue()
    if closed_segment:
        assert "logical response incomplete" in stdout.getvalue()
        assert "INVALIDATED" not in stdout.getvalue()
    else:
        assert "INVALIDATED" in stdout.getvalue()
    assert "COMPLETED" not in stdout.getvalue()
