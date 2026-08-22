#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI command surface checks."""

import os
import subprocess
import sys
from pathlib import Path

from linktools.commands.ai.acp import command as acp_command
from linktools.commands.ai.run import command as run_command
from linktools.cli.argparse import ConfigAction


def test_ai_run_exposes_model_configuration_but_not_asset_storage() -> None:
    parser = run_command.create_parser()
    actions = {action.dest: action for action in parser._actions}
    assert all(isinstance(actions[name], ConfigAction) for name in ("api_key", "base_url", "model"))
    assert not any(action.dest in {"assets", "asset_root", "asset_store", "storage_root"} for action in parser._actions)


def test_ai_run_storage_backend_defaults_to_sqlite() -> None:
    parser = run_command.create_parser()
    assert parser.parse_args(["prompt"]).storage == "sqlite"
    assert parser.parse_args(["prompt", "--storage", "filesystem"]).storage == "filesystem"


def test_ai_acp_memory_scope_defaults_to_workspace() -> None:
    args = acp_command.create_parser().parse_args([])
    assert args.memory is None
    assert acp_command.create_parser().parse_args(["--memory", "custom"]).memory == "custom"


def test_ai_asset_command_is_removed() -> None:
    environment = dict(os.environ)
    source_root = Path(__file__).parents[2]
    environment["PYTHONPATH"] = os.pathsep.join((str(source_root / "linktools-ai/src"), str(source_root / "linktools/src")))
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
