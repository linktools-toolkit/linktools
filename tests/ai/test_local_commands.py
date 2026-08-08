#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Workspace application and command-surface checks."""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

from pydantic_ai.models.test import TestModel

from linktools.ai.agent import WorkspaceAgentRunner
from linktools.ai.app import RuntimePersistenceConfig
from linktools.ai.app import open_workspace_runtime
from linktools.ai.core import ErrorCode, AIError
from linktools.ai.workspace import Workspace
from linktools.commands.ai.run import command as run_command


def test_ai_run_executes_the_workspace_test_model(tmp_path: Path, capsys, monkeypatch) -> None:
    parser = run_command.create_parser()
    args = parser.parse_args(["--model", "test", "hello"])
    monkeypatch.chdir(tmp_path)
    assert run_command.run(args) == 0
    assert "success (no tool calls)" in capsys.readouterr().out


def test_workspace_runtime_uses_database_storage(tmp_path: Path) -> None:
    async def run() -> str:
        workspace = Workspace.load(tmp_path)
        runner = WorkspaceAgentRunner(workspace.root, workspace.config, model=TestModel())
        config = RuntimePersistenceConfig.sqlite(str(tmp_path / "runtime.db"), namespace=workspace.workspace_id, deployment_id="workspace")
        async with open_workspace_runtime(workspace, config=config, runner=runner) as runtime:
            result = await runtime.run("main", "hello", idempotency_key="database-key")
            return result.output

    assert asyncio.run(run()) == "success (no tool calls)"


def test_workspace_runtime_rejects_reused_key_after_head_changes(tmp_path: Path) -> None:
    async def run() -> None:
        workspace = Workspace.load(tmp_path)
        runner = WorkspaceAgentRunner(workspace.root, workspace.config, model=TestModel())
        async with open_workspace_runtime(workspace, config=RuntimePersistenceConfig.in_memory(namespace=workspace.workspace_id), runner=runner) as runtime:
            await runtime.run("main", "one", idempotency_key="same")
            try:
                await runtime.run("main", "two", idempotency_key="same")
            except AIError as error:
                assert error.code is ErrorCode.IDEMPOTENCY_CONFLICT
            else:
                raise AssertionError("idempotency conflict was not rejected")

    asyncio.run(run())


def test_ai_asset_command_is_removed() -> None:
    environment = dict(os.environ)
    source_root = Path(__file__).parents[2]
    environment["PYTHONPATH"] = os.pathsep.join((str(source_root / "linktools-ai/src"), str(source_root / "linktools/src")))
    environment["DEBUG"] = "false"
    result = subprocess.run([sys.executable, "-m", "linktools", "ai", "asset", "--help"], env=environment, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "invalid choice: 'asset'" in result.stderr
