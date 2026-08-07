#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Executable local run and command-surface checks."""

import os
import subprocess
import sys
import asyncio
from datetime import datetime, timezone
from pathlib import Path

from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from linktools.ai.agent.runner import LocalAgentRunner
from linktools.ai.core.json import JsonValue
from linktools.ai.core.errors import ErrorCode, LinktoolsAIError
from linktools.ai.capability.tool import ToolOperationRecord
from linktools.ai.core.value import ToolOperationStatus
from linktools.ai.local import LocalAgentRuntime, LocalExecutionRecord, LocalProject, LocalRecordStore, LocalRunResult, build_file_runtime
from linktools.ai.local.tool import build_local_tools
from linktools.commands.ai.run import command as run_command


def test_ai_run_executes_the_local_test_model(tmp_path: Path, capsys, monkeypatch) -> None:
    parser = run_command.create_parser()
    args = parser.parse_args(
        ["--model", "test", "--base-url", "", "--api-key", "", "hello"]
    )

    monkeypatch.chdir(tmp_path)
    assert run_command.run(args) == 0
    assert "success (no tool calls)" in capsys.readouterr().out
    assert (tmp_path / ".linktools/sessions/main.json").is_file()


def test_ai_run_separates_work_and_runtime_storage(tmp_path: Path, capsys, monkeypatch) -> None:
    work_root = tmp_path / "work"
    storage_root = tmp_path / "runtime"
    work_root.mkdir()
    parser = run_command.create_parser()
    args = parser.parse_args(
        [
            "--model",
            "test",
            "--base-url",
            "",
            "--api-key",
            "",
            "--project",
            str(work_root),
            "--storage",
            str(storage_root),
            "hello",
        ]
    )

    monkeypatch.chdir(tmp_path)
    assert run_command.run(args) == 0
    capsys.readouterr()
    assert (storage_root / ".linktools/sessions/main.json").is_file()
    assert (storage_root / ".linktools/records").is_dir()
    assert not (work_root / ".linktools/sessions/main.json").exists()


def test_local_runtime_forwards_each_text_delta(tmp_path: Path) -> None:
    async def stream_function(messages, info):
        yield "first "
        await asyncio.sleep(0)
        yield "second"

    async def run() -> list[str]:
        project = LocalProject.discover(tmp_path)
        runtime = LocalAgentRuntime(
            project,
            runner=LocalAgentRunner(project.root, project.config, model=FunctionModel(stream_function=stream_function)),
        )
        chunks: list[str] = []

        async def on_text(value: str) -> None:
            chunks.append(value)

        await runtime.run("main", "hello", on_text=on_text)
        return chunks

    assert asyncio.run(run()) == ["first ", "second"]


def test_local_runtime_emits_tool_lifecycle_events(tmp_path: Path) -> None:
    async def run() -> list[dict[str, JsonValue]]:
        project = LocalProject.discover(tmp_path)
        runtime = LocalAgentRuntime(
            project,
            runner=LocalAgentRunner(project.root, project.config, model=TestModel(), tools=build_local_tools(project.root)),
        )
        events: list[dict[str, JsonValue]] = []
        await runtime.run("main", "inspect", on_event=events.append)
        return events

    events = asyncio.run(run())
    starts = [event["name"] for event in events if event["type"] == "tool" and event["phase"] == "start"]
    ends = [event for event in events if event["type"] == "tool" and event["phase"] == "end"]
    assert starts == ["list_dir", "read_file", "write_file", "bash"]
    assert len(ends) == len(starts)


def test_local_runtime_emits_thinking_events(tmp_path: Path) -> None:
    async def stream_function(messages, info):
        yield "answer"

    async def run() -> list[dict[str, JsonValue]]:
        project = LocalProject.discover(tmp_path)
        runtime = LocalAgentRuntime(
            project,
            runner=LocalAgentRunner(project.root, project.config, model=FunctionModel(stream_function=stream_function)),
        )
        events: list[dict[str, JsonValue]] = []
        await runtime.run("main", "hello", on_event=events.append)
        return events

    assert asyncio.run(run()) == [
        {"type": "text", "text": "answer"},
        {"type": "text_end"},
    ]


def test_local_runtime_idempotency_and_observer_isolation(tmp_path: Path) -> None:
    calls = 0

    async def stream_function(messages, info):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        yield "answer"

    async def run() -> tuple[LocalRunResult, LocalRunResult]:
        project = LocalProject.discover(tmp_path)
        runtime = LocalAgentRuntime(
            project,
            runner=LocalAgentRunner(project.root, project.config, model=FunctionModel(stream_function=stream_function)),
        )

        def broken_observer(event: dict[str, JsonValue]) -> None:
            raise RuntimeError("observer failure")

        first, second = await asyncio.gather(
            runtime.run("main", "hello", idempotency_key="same", on_event=broken_observer),
            runtime.run("main", "hello", idempotency_key="same"),
        )
        return first, second

    first, second = asyncio.run(run())
    assert first.execution_id == second.execution_id
    assert calls == 1

    async def conflict() -> None:
        project = LocalProject.discover(tmp_path)
        runtime = LocalAgentRuntime(
            project,
            runner=LocalAgentRunner(project.root, project.config, model=FunctionModel(stream_function=lambda messages, info: _answer())),
        )
        await runtime.run("main", "one", idempotency_key="conflict")
        try:
            await runtime.run("main", "two", idempotency_key="conflict")
        except LinktoolsAIError as error:
            assert error.code == ErrorCode.IDEMPOTENCY_CONFLICT
        else:
            raise AssertionError("idempotency conflict was not rejected")

    async def _answer():
        yield "answer"

    asyncio.run(conflict())


def test_local_record_store_marks_interrupted_runs_after_restart(tmp_path: Path) -> None:
    async def run() -> LocalExecutionRecord | None:
        timestamp = datetime.now(timezone.utc)
        store = LocalRecordStore(tmp_path, "project")
        await store.save(
            LocalExecutionRecord(
                "project",
                "session",
                0,
                str(tmp_path),
                "execution",
                "STARTED",
                timestamp,
                timestamp,
                None,
            )
        )
        restarted = LocalRecordStore(tmp_path, "project")
        return await restarted.get("execution")

    record = asyncio.run(run())
    assert record is not None
    assert record.status == "CANCELLED"
    assert record.stop_reason == "PROCESS_RESTARTED"


def test_local_tool_state_survives_restart(tmp_path: Path) -> None:
    async def run() -> ToolOperationRecord | None:
        timestamp = datetime.now(timezone.utc)
        first = build_file_runtime(str(tmp_path), project_id="project", local_tenant_id="project")
        await first.initialize()
        await first.persistence.tools.reserve(
            ToolOperationRecord(
                "operation", "project", "run", "call", "idempotency", "tool", "arguments", "binding", True,
                    ToolOperationStatus.PENDING, None, 0, None, None, None, None, timestamp, timestamp,
            )
        )
        await first.close()
        second = build_file_runtime(str(tmp_path), project_id="project", local_tenant_id="project")
        await second.initialize()
        value = await second.persistence.tools.get_operation("operation", tenant_id="project")
        await second.close()
        return value

    assert asyncio.run(run()) is not None


def test_ai_asset_command_is_removed() -> None:
    environment = dict(os.environ)
    source_root = Path(__file__).parents[2]
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(source_root / "linktools-ai/src"), str(source_root / "linktools/src"))
    )
    environment["DEBUG"] = "false"
    result = subprocess.run(
        [sys.executable, "-m", "linktools", "ai", "asset", "--help"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "invalid choice: 'asset'" in result.stderr
