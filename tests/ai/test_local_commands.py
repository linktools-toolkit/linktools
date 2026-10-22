#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI command surface checks."""

import os
import subprocess
import sys
from argparse import Namespace
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from linktools.ai.workspace import Workspace
from linktools.cli import CommandError
from linktools.cli.argparse import ConfigAction
import linktools.commands.ai.acp as acp_module
from linktools.commands.ai._common import _local_runtime_models
from linktools.commands.ai.acp import command as acp_command
from linktools.commands.ai.run import command as run_command


def test_ai_command_group_exposes_only_execution_acp_and_web() -> None:
    from linktools.cli import iter_module_commands
    import linktools.commands.ai as ai_commands

    names = {entry.command.name for entry in iter_module_commands(ai_commands) if entry.command is not None}
    assert names == {"run", "acp", "web"}


def test_ai_local_commands_share_runtime_arguments() -> None:
    common = {"project", "api_key", "base_url", "model", "memory"}
    for command in (run_command, acp_command):
        parser = command.create_parser()
        actions = {action.dest: action for action in parser._actions}
        assert common <= set(actions)
        assert "vision" not in actions
        assert all(
            isinstance(actions[name], ConfigAction)
            for name in ("model_api",)
        )
        assert not any(
            action.dest
            in {"assets", "asset_root", "asset_store", "storage", "storage_root"}
            for action in parser._actions
        )


def test_ai_local_memory_scope_argument_is_consistent() -> None:
    cases = (
        (run_command, ["prompt"], ["prompt", "--memory", "custom"]),
        (acp_command, [], ["--memory", "custom"]),
    )
    for command, default_args, custom_args in cases:
        assert command.create_parser().parse_args(default_args).memory is None
        assert command.create_parser().parse_args(custom_args).memory == "custom"


def test_ai_acp_uses_shared_local_runtime_composition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import linktools.ai.acp as acp_adapter

    workspace = Workspace.initialize(tmp_path)
    models = object()
    opened: list[tuple[Workspace, object]] = []

    monkeypatch.setattr(acp_module, "_load_workspace", lambda _root: workspace)
    monkeypatch.setattr(
        acp_module,
        "_local_runtime_models",
        lambda _workspace, _args: models,
    )

    @asynccontextmanager
    async def open_local_runtime(
        runtime_workspace: Workspace,
        *,
        models: object,
    ):
        opened.append((runtime_workspace, models))
        yield SimpleNamespace(default_principal=object())

    monkeypatch.setattr(acp_module, "_open_local_runtime", open_local_runtime)
    monkeypatch.setattr(acp_adapter, "ACPAgent", lambda *_args, **_kwargs: object())

    async def serve_stdio(_agent: object) -> None:
        return None

    monkeypatch.setattr(acp_adapter, "serve_stdio", serve_stdio)

    assert acp_command.run(acp_command.create_parser().parse_args([])) == 0
    assert opened == [(workspace, models)]


def test_ai_local_runtime_models_fall_back_to_workspace_model(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path, {"model": "workspace/model"})
    args = Namespace(model_api=None, model=None, base_url=None, api_key=None)

    binding = _local_runtime_models(workspace, args).capture().resolve("default")

    assert binding.model_identity == "openai:workspace/model"


def test_ai_run_translates_invalid_model_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path, {})
    monkeypatch.setattr(
        "linktools.commands.ai.run._load_workspace",
        lambda _root: workspace,
    )
    args = run_command.create_parser().parse_args(
        ["hello", "--model", "testing/model", "--base-url", "not-a-url"]
    )

    with pytest.raises(CommandError):
        run_command.run(args)


def test_ai_cli_rejects_unknown_commands() -> None:
    environment = dict(os.environ)
    source_root = Path(__file__).parents[2]
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(source_root / "linktools-ai/src"), str(source_root / "linktools/src"))
    )
    environment["DEBUG"] = "false"
    result = subprocess.run(
        [sys.executable, "-m", "linktools", "ai", "unknown-command", "--help"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "invalid choice: 'unknown-command'" in result.stderr


@pytest.mark.parametrize(
    ("arguments", "expected_output"),
    (
        (("--help",), "ai"),
        (("ai", "run", "--help"), "prompt"),
        (("ai", "acp", "--help"), "ai acp"),
        (("ai", "web", "--help"), "--port"),
    ),
)
def test_lightweight_cli_commands_do_not_load_execution_dependencies(
    arguments: tuple[str, ...],
    expected_output: str,
    tmp_path: Path,
) -> None:
    environment = dict(os.environ)
    source_root = Path(__file__).parents[2]
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(source_root / "linktools-ai/src"), str(source_root / "linktools/src"))
    )
    environment["LINKTOOLS_PATH"] = str(tmp_path / "home")
    environment["DEBUG"] = "false"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import runpy
import sys

sys.argv[0] = "lt"
try:
    runpy.run_module("linktools", run_name="__main__")
except SystemExit as error:
    assert error.code == 0, error.code

heavy_dependencies = (
    "linktools.ai.runtime",
    "linktools.ai.capability",
    "linktools.ai.model",
    "linktools.ai.acp",
    "pydantic_ai",
    "fastmcp",
    "openai",
)
loaded = tuple(
    name for name in sys.modules
    if any(name == prefix or name.startswith(prefix + ".") for prefix in heavy_dependencies)
)
assert not loaded, loaded
""",
            *arguments,
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert expected_output in result.stdout


def test_local_input_types_expand_workspace_categories(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path, {
        "model": "opaque-model",
        "input_types": ["audio", "mime:application/x-private", "mime:image/*"],
    })
    binding = _local_runtime_models(workspace, Namespace(model_api=None, model=None, base_url=None, api_key=None)).capture().resolve("default")
    from typing import get_args
    from linktools.ai.model import AudioMediaType

    assert binding.contract["input_mime_types"] == sorted({*get_args(AudioMediaType), "application/x-private", "image/*"})
    assert "vision" not in binding.contract
    assert "inputs" not in binding.contract


def test_local_input_default_rejects_attachments(tmp_path: Path) -> None:
    binding = _local_runtime_models(Workspace(tmp_path, {}), Namespace(model_api=None, model="opaque", base_url=None, api_key=None)).capture().resolve("default")
    assert binding.contract["input_mime_types"] == []


@pytest.mark.parametrize("source", ["OPENAI_VISION", "LINKTOOLS_OPENAI_VISION", "workspace", "config"])
@pytest.mark.parametrize("value", ["true", "false", ""])
def test_local_input_types_ignore_unrelated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: str, value: str) -> None:
    from linktools.core import environ

    config = environ.build_config("migration-test", "LINKTOOLS_")
    monkeypatch.setattr(environ, "config", config)
    workspace = Workspace(tmp_path, {})
    if source == "workspace":
        workspace.config["vision"] = value
    elif source == "config":
        config.set("OPENAI_VISION", value)
    else:
        monkeypatch.setenv(source, value)
    workspace.config["input_types"] = ["mime:application/pdf"]
    binding = _local_runtime_models(workspace, Namespace(model_api=None, model="opaque", base_url=None, api_key=None)).capture().resolve("default")
    assert binding.contract["input_mime_types"] == ["application/pdf"]


@pytest.mark.parametrize("config", [
    {"input_types": None}, {"input_types": {}},
    {"input_types": "audio"}, {"input_types": [123]},
    {"input_types": ["text"]}, {"input_types": ["image/png"]},
    {"input_types": ["mime:*/*"]}, {"input_types": ["mime:"]},
    {"input_types": [{"mime_type": "image/png"}]},
])
def test_local_input_invalid_configuration_fails_closed(tmp_path: Path, config: dict) -> None:
    from linktools.ai.errors import AIError, ErrorCode

    with pytest.raises(AIError) as raised:
        _local_runtime_models(Workspace(tmp_path, config), Namespace(model_api=None, model="opaque", base_url=None, api_key=None))
    assert raised.value.code is ErrorCode.MODEL_CONFIG_INVALID


@pytest.mark.parametrize("key", ["input_capabilities", "unknown_inputs", "input_mime_types"])
def test_local_model_reads_only_input_types(tmp_path: Path, key: str) -> None:
    workspace = Workspace(tmp_path, {key: ["image/png"]})
    binding = _local_runtime_models(workspace, Namespace(model_api=None, model="opaque", base_url=None, api_key=None)).capture().resolve("default")
    assert binding.contract["input_mime_types"] == []


def test_local_missing_model_keeps_dependency_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from linktools.ai.errors import AIError, ErrorCode

    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with pytest.raises(AIError) as raised:
        _local_runtime_models(Workspace(tmp_path, {}), Namespace(model_api=None, model=None, base_url=None, api_key=None))
    assert raised.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
