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
from linktools.ai.errors import ErrorCode, AIError
from linktools.ai.workspace import Workspace
from linktools.cli.argparse import ConfigAction
from linktools.commands.ai.run import command as run_command
from tests.ai.persistence.helper import _open_sql_workspace


def test_ai_run_executes_the_workspace_test_model(tmp_path: Path, capsys, monkeypatch) -> None:
    parser = run_command.create_parser()
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    args = parser.parse_args(["--model", "test", "--memory-namespace", "test", "hello"])
    actions = {action.dest: action for action in parser._actions}
    assert all(isinstance(actions[name], ConfigAction) for name in ("api_key", "base_url", "model"))
    assert args.model == "test"
    monkeypatch.chdir(tmp_path)
    assert run_command.run(args) == 0
    assert "success (no tool calls)" in capsys.readouterr().out


def test_workspace_runtime_uses_database_storage(tmp_path: Path) -> None:
    async def run() -> str:
        workspace = Workspace.load(tmp_path)
        runner = WorkspaceAgentRunner(workspace.root, workspace.config, model=TestModel())
        config = RuntimePersistenceConfig.sqlite(str(tmp_path / "runtime.db"), namespace=workspace.workspace_id, deployment_id="workspace")
        async with _open_sql_workspace(workspace, config, runner=runner) as runtime:
            result = await runtime.run("main", "hello", idempotency_key="database-key", memory_namespace="test")
            return result.output

    assert asyncio.run(run()) == "success (no tool calls)"


def test_workspace_runtime_rejects_reused_key_after_head_changes(tmp_path: Path) -> None:
    async def run() -> None:
        workspace = Workspace.load(tmp_path)
        runner = WorkspaceAgentRunner(workspace.root, workspace.config, model=TestModel())
        async with open_workspace_runtime(workspace, config=RuntimePersistenceConfig.in_memory(namespace=workspace.workspace_id), runner=runner) as runtime:
            await runtime.run("main", "one", idempotency_key="same", memory_namespace="test")
            try:
                await runtime.run("main", "two", idempotency_key="same", memory_namespace="test")
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
