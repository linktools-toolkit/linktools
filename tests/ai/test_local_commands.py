#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI command surface checks."""

import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from linktools.ai.workspace import Workspace
import linktools.commands.ai._common as common_command
from linktools.cli import CommandError
from linktools.cli.argparse import ConfigAction
from linktools.commands.ai._common import _ModelOptions
from linktools.commands.ai.acp import command as acp_command
from linktools.commands.ai.run import command as run_command


def test_ai_command_group_exposes_only_execution_acp_and_web() -> None:
    from linktools.cli import iter_module_commands
    import linktools.commands.ai as ai_commands

    names = {entry.command.name for entry in iter_module_commands(ai_commands) if entry.command is not None}
    assert names == {"run", "acp", "web"}


def test_ai_local_commands_share_runtime_arguments() -> None:
    common = {"project", "api_key", "base_url", "model"}
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


def test_ai_local_commands_do_not_expose_memory_scope() -> None:
    cases = (
        (run_command, ["prompt"], ["prompt", "--memory", "custom"]),
        (acp_command, [], ["--memory", "custom"]),
    )
    for command, default_args, custom_args in cases:
        assert "memory" not in vars(command.create_parser().parse_args(default_args))
        with pytest.raises(SystemExit):
            command.create_parser().parse_args(custom_args)


def test_ai_acp_serves_with_the_shared_local_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import linktools.ai.acp as acp_adapter
    from ._runtime_test_helpers import RuntimeUsageModels

    monkeypatch.setattr(common_command._ModelOptions, "create_model", lambda _self: RuntimeUsageModels())
    monkeypatch.setattr(acp_adapter, "ACPAgent", lambda runtime, **_kwargs: runtime)

    async def serve_stdio(runtime) -> None:
        result = await runtime.agents.get().run("hello")
        assert result.result.output == {"text": "done"}

    monkeypatch.setattr(acp_adapter, "serve_stdio", serve_stdio)

    args = acp_command.create_parser().parse_args(["--project", str(tmp_path)])
    assert acp_command.run(args) == 0


@pytest.mark.parametrize("module,hint", [
    ("linktools.ai.acp", "agent-client-protocol"), ("uvicorn", "linktools-ai[web]"),
])
def test_optional_command_dependencies_fail_before_opening_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, module: str, hint: str,
) -> None:
    from linktools.commands.ai.web import command as web_command

    monkeypatch.setitem(sys.modules, module, None)
    command = acp_command if module == "linktools.ai.acp" else web_command
    args = command.create_parser().parse_args(["--project", str(tmp_path)])
    with pytest.raises(CommandError) as error:
        command.run(args)
    assert hint in str(error.value)
    assert not (tmp_path / ".linktools" / "runtime").exists()


def test_ai_run_uses_shared_runtime_to_emit_json_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    from ._runtime_test_helpers import RuntimeUsageModels

    monkeypatch.setattr(common_command._ModelOptions, "create_model", lambda _self: RuntimeUsageModels())
    args = run_command.create_parser().parse_args(["hello", "--project", str(tmp_path), "--json"])
    assert run_command.run(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "SUCCEEDED"
    assert result["output"] == {"text": "done"}


def test_ai_model_falls_back_to_workspace_model(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path, {"model": "workspace/model"})
    args = Namespace(model_api=None, model=None, base_url=None, api_key=None)

    binding = _ModelOptions.load(workspace, args).create_model()

    assert binding.model_identity == "openai:workspace/model"


def test_ai_run_translates_invalid_model_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path, {})
    monkeypatch.setattr(
        "linktools.commands.ai._common._load_workspace",
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
    binding = _ModelOptions.load(workspace, Namespace(model_api=None, model=None, base_url=None, api_key=None)).create_model()
    from typing import get_args
    from linktools.ai.model import AudioMediaType

    assert binding.contract["input_mime_types"] == sorted({*get_args(AudioMediaType), "application/x-private", "image/*"})
    assert "vision" not in binding.contract
    assert "inputs" not in binding.contract


def test_local_input_default_rejects_attachments(tmp_path: Path) -> None:
    binding = _ModelOptions.load(Workspace(tmp_path, {}), Namespace(model_api=None, model="opaque", base_url=None, api_key=None)).create_model()
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
    binding = _ModelOptions.load(workspace, Namespace(model_api=None, model="opaque", base_url=None, api_key=None)).create_model()
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
        _ModelOptions.load(Workspace(tmp_path, config), Namespace(model_api=None, model="opaque", base_url=None, api_key=None)).create_model()
    assert raised.value.code is ErrorCode.MODEL_CONFIG_INVALID


@pytest.mark.parametrize("key", ["input_capabilities", "unknown_inputs", "input_mime_types"])
def test_local_model_reads_only_input_types(tmp_path: Path, key: str) -> None:
    workspace = Workspace(tmp_path, {key: ["image/png"]})
    binding = _ModelOptions.load(workspace, Namespace(model_api=None, model="opaque", base_url=None, api_key=None)).create_model()
    assert binding.contract["input_mime_types"] == []


def test_local_missing_model_keeps_dependency_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from linktools.ai.errors import AIError, ErrorCode

    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with pytest.raises(AIError) as raised:
        _ModelOptions.load(Workspace(tmp_path, {}), Namespace(model_api=None, model=None, base_url=None, api_key=None)).create_model()
    assert raised.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
