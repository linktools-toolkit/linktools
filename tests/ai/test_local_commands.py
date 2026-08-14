#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Workspace application and command-surface checks."""

import hashlib
import os
import subprocess
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from linktools.ai.model import OpenAIModelMaterializer
from linktools.ai.spec import AgentSpec, AgentSpecCodec, PromptSpec, PromptSpecCodec
from linktools.ai.workspace import Workspace
from pydantic_ai import RunContext
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models import ModelRequestParameters, ModelSettings, StreamedResponse
from linktools.cli.argparse import ConfigAction
from linktools.commands.ai.acp import command as acp_command
from linktools.commands.ai.run import command as run_command
from pydantic_ai.models.test import TestModel


class _RecordingTestModel(TestModel):
    def __init__(self, output: str) -> None:
        super().__init__(call_tools=[], custom_output_text=output)
        self.requests: list[list[ModelMessage]] = []

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: "RunContext[object] | None" = None,
    ) -> AsyncIterator[StreamedResponse]:
        self.requests.append(list(messages))
        async with super().request_stream(messages, model_settings, model_request_parameters, run_context) as response:
            yield response



def test_ai_run_executes_the_workspace_test_model(tmp_path: Path, capsys, monkeypatch) -> None:
    _write_run_assets(tmp_path)
    parser = run_command.create_parser()
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    args = parser.parse_args(["--model", "test", "hello"])
    actions = {action.dest: action for action in parser._actions}
    assert all(isinstance(actions[name], ConfigAction) for name in ("api_key", "base_url", "model"))
    assert args.model == "test"
    assert not any(action.dest == "session" for action in parser._actions)
    assert not any(action.dest in {"storage", "assets", "agent", "memory"} for action in parser._actions)
    monkeypatch.setattr(OpenAIModelMaterializer, "materialize", lambda self, route, connection: TestModel(call_tools=[], custom_output_text="success"))
    monkeypatch.chdir(tmp_path)
    assert run_command.run(args) == 0
    assert capsys.readouterr().out.strip() == "success"
    namespace_root = tmp_path / ".linktools" / "runtime" / hashlib.sha256(Workspace.load(tmp_path).workspace_id.encode("utf-8")).hexdigest()
    assert (namespace_root / "manifest.json").is_file()


def test_ai_run_uses_default_specs_without_asset_files(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    args = run_command.create_parser().parse_args(["--project", str(tmp_path), "--model", "test", "hello"])
    monkeypatch.setattr(
        OpenAIModelMaterializer,
        "materialize",
        lambda self, route, connection: TestModel(call_tools=[], custom_output_text="fallback"),
    )

    assert run_command.run(args) == 0
    assert capsys.readouterr().out.strip() == "fallback"


def test_ai_run_loads_agent_spec_from_asset_store(tmp_path: Path, capsys, monkeypatch) -> None:
    asset_path = tmp_path / ".linktools" / "agent" / "default"
    asset_path.parent.mkdir(parents=True)
    asset_path.write_bytes(
        AgentSpecCodec().encode(
            AgentSpec("default", 1, "default", (), "assistant-text", 1, ("Use the asset spec.",))
        )
    )
    prompt_path = tmp_path / ".linktools" / "prompt" / "default"
    prompt_path.parent.mkdir(parents=True)
    prompt_path.write_bytes(PromptSpecCodec().encode(PromptSpec("default", 1, "system", ())))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    args = run_command.create_parser().parse_args(["--project", str(tmp_path), "--model", "test", "hello"])
    monkeypatch.setattr(OpenAIModelMaterializer, "materialize", lambda self, route, connection: TestModel(call_tools=[], custom_output_text="asset"))

    assert run_command.run(args) == 0
    assert capsys.readouterr().out.strip() == "asset"


def test_ai_run_reuses_workspace_session_after_cold_restart(tmp_path: Path, capsys, monkeypatch) -> None:
    _write_run_assets(tmp_path)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    models: list[_RecordingTestModel] = []

    def materialize(self: OpenAIModelMaterializer, route: object, connection: object) -> _RecordingTestModel:
        del self, route, connection
        model = _RecordingTestModel(f"response-{len(models) + 1}")
        models.append(model)
        return model

    monkeypatch.setattr(OpenAIModelMaterializer, "materialize", materialize)
    monkeypatch.chdir(tmp_path)
    args = run_command.create_parser().parse_args(["--model", "test", "first prompt"])
    assert run_command.run(args) == 0
    assert capsys.readouterr().out.strip() == "response-1"

    args = run_command.create_parser().parse_args(["--model", "test", "second prompt"])
    assert run_command.run(args) == 0
    assert capsys.readouterr().out.strip() == "response-2"

    assert len(models) == 2
    assert any(
        isinstance(part, UserPromptPart) and part.content == "first prompt"
        for message in models[1].requests[0]
        if isinstance(message, ModelRequest)
        for part in message.parts
    )


def _write_run_assets(root: Path) -> None:
    asset_root = root / ".linktools"
    (asset_root / "agent").mkdir(parents=True)
    (asset_root / "prompt").mkdir(parents=True)
    (asset_root / "agent" / "default").write_bytes(
        AgentSpecCodec().encode(AgentSpec("default", 1, "default", (), "assistant-text", 1))
    )
    (asset_root / "prompt" / "default").write_bytes(
        PromptSpecCodec().encode(PromptSpec("default", 1, "system", ()))
    )


def test_ai_acp_memory_namespace_defaults_to_workspace() -> None:
    args = acp_command.create_parser().parse_args([])
    assert args.memory is None
    explicit = acp_command.create_parser().parse_args(["--memory", "custom"])
    assert explicit.memory == "custom"


def test_ai_asset_command_is_removed() -> None:
    environment = dict(os.environ)
    source_root = Path(__file__).parents[2]
    environment["PYTHONPATH"] = os.pathsep.join((str(source_root / "linktools-ai/src"), str(source_root / "linktools/src")))
    environment["DEBUG"] = "false"
    result = subprocess.run([sys.executable, "-m", "linktools", "ai", "asset", "--help"], env=environment, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "invalid choice: 'asset'" in result.stderr
