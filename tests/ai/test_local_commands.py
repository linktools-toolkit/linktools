#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI command surface checks."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from linktools.cli.argparse import ConfigAction
import linktools.commands.ai.acp as acp_module
from linktools.commands.ai.acp import command as acp_command
from linktools.commands.ai.run import command as run_command
from linktools.ai.workspace import Workspace


def test_ai_run_exposes_model_configuration_but_not_storage_selection() -> None:
    parser = run_command.create_parser()
    actions = {action.dest: action for action in parser._actions}
    assert all(
        isinstance(actions[name], ConfigAction)
        for name in ("api_key", "base_url", "model")
    )
    assert not any(
        action.dest
        in {"assets", "asset_root", "asset_store", "storage", "storage_root"}
        for action in parser._actions
    )


def test_ai_acp_memory_scope_defaults_to_workspace() -> None:
    args = acp_command.create_parser().parse_args([])
    assert args.memory is None
    assert acp_command.create_parser().parse_args(["--memory", "custom"]).memory == "custom"


def test_ai_acp_uses_shared_local_runtime_composition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path, workspace_id="workspace")
    state = object()
    metrics = object()
    opened: dict[str, object] = {}

    monkeypatch.setattr(acp_module, "_load_workspace", lambda _root: workspace)
    monkeypatch.setattr(acp_module, "_local_runtime_state", lambda _workspace: state)

    async def local_metrics(_workspace: Workspace) -> object:
        return metrics

    monkeypatch.setattr(acp_module, "_local_metrics", local_metrics)

    class RuntimeContext:
        async def __aenter__(self) -> SimpleNamespace:
            return SimpleNamespace(default_principal=object())

        async def __aexit__(
            self,
            exc_type: object,
            exc_value: object,
            traceback: object,
        ) -> None:
            return None

    class LocalRuntime:
        @staticmethod
        def open(
            runtime_workspace: Workspace,
            *,
            state: object,
            metrics: object,
        ) -> RuntimeContext:
            opened.update(
                workspace=runtime_workspace,
                state=state,
                metrics=metrics,
            )
            return RuntimeContext()

    monkeypatch.setattr(acp_module, "Runtime", LocalRuntime)
    monkeypatch.setattr(acp_module, "ACPAgent", lambda *_args, **_kwargs: object())

    async def serve_stdio(_agent: object) -> None:
        return None

    monkeypatch.setattr(acp_module, "serve_stdio", serve_stdio)

    assert acp_command.run(acp_command.create_parser().parse_args([])) == 0
    assert opened == {
        "workspace": workspace,
        "state": state,
        "metrics": metrics,
    }


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
