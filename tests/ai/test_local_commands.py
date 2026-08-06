#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Executable local run and command-surface checks."""

import os
import subprocess
import sys
import asyncio
from pathlib import Path

from pydantic_ai.models.function import DeltaThinkingPart, FunctionModel
from pydantic_ai.models.test import TestModel

from linktools.ai.local import LocalAgentRuntime, LocalProject
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


def test_local_runtime_forwards_each_text_delta(tmp_path: Path) -> None:
    async def stream_function(messages, info):
        yield "first "
        await asyncio.sleep(0)
        yield "second"

    async def run() -> list[str]:
        runtime = LocalAgentRuntime(
            LocalProject.discover(tmp_path),
            model=FunctionModel(stream_function=stream_function),
        )
        chunks: list[str] = []

        async def on_text(value: str) -> None:
            chunks.append(value)

        await runtime.run("main", "hello", on_text=on_text)
        return chunks

    assert asyncio.run(run()) == ["first ", "second"]


def test_local_runtime_emits_tool_lifecycle_events(tmp_path: Path) -> None:
    async def run() -> list[dict[str, object]]:
        runtime = LocalAgentRuntime(LocalProject.discover(tmp_path), model=TestModel())
        events: list[dict[str, object]] = []
        await runtime.run("main", "inspect", on_event=events.append)
        return events

    events = asyncio.run(run())
    starts = [event["name"] for event in events if event["type"] == "tool" and event["phase"] == "start"]
    ends = [event for event in events if event["type"] == "tool" and event["phase"] == "end"]
    assert starts == ["list_dir", "read_file", "write_file", "bash"]
    assert len(ends) == len(starts)


def test_local_runtime_emits_thinking_events(tmp_path: Path) -> None:
    async def stream_function(messages, info):
        yield {0: DeltaThinkingPart(content="private thought")}
        yield "answer"

    async def run() -> list[dict[str, object]]:
        runtime = LocalAgentRuntime(
            LocalProject.discover(tmp_path),
            model=FunctionModel(stream_function=stream_function),
            tools=(),
        )
        events: list[dict[str, object]] = []
        await runtime.run("main", "hello", on_event=events.append)
        return events

    assert asyncio.run(run()) == [
        {"type": "thinking", "text": "private thought"},
        {"type": "text", "text": "answer"},
        {"type": "text_end"},
    ]


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
