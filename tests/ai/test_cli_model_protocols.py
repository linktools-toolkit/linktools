#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local CLI protocol selection uses isolated configuration and native adapters."""

import json
import os
import subprocess
import sys

import httpx2
import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models import ModelRequestParameters

from linktools.ai.workspace import Workspace
from linktools.commands.ai._common import _local_model_options, _local_runtime_models
from linktools.commands.ai.run import command as run_command
from linktools.commands.ai.acp import command as acp_command
from linktools.commands.ai.web import command as web_command
from tests.ai.test_model_protocols import response


@pytest.mark.parametrize("command,arguments", [(run_command, ["hello"]), (acp_command, []), (web_command, [])])
@pytest.mark.parametrize("api,prefix", [("openai-chat", "OPENAI"), ("openai-responses", "OPENAI"),
                                       ("anthropic-messages", "ANTHROPIC")])
def test_cli_protocol_configuration_round_trip(monkeypatch, tmp_path, command, arguments, api, prefix):
    config = command.environ.build_config("cli-protocol", "LINKTOOLS_")
    monkeypatch.setattr(command.environ, "config", config)
    for provider in ("OPENAI", "ANTHROPIC"):
        monkeypatch.setenv(f"{provider}_MODEL", f"{provider.lower()}-model")
        monkeypatch.setenv(f"{provider}_API_KEY", f"{provider.lower()}-fixture-key")
        monkeypatch.setenv(f"{provider}_BASE_URL", f"https://{provider.lower()}.invalid/v1")
    root = tmp_path / ".linktools"
    root.mkdir()
    (root / "config.yaml").write_text(f"model_api: {api}\nmodel: workspace-model\n")
    workspace = Workspace.discover(tmp_path)
    args = command.create_parser().parse_args(arguments)
    options = _local_model_options(workspace, args)
    assert options == {"api": api, "model": f"{prefix.lower()}-model",
                       "api_key": f"{prefix.lower()}-fixture-key", "base_url": f"https://{prefix.lower()}.invalid/v1"}
    binding = _local_runtime_models(workspace, args).capture().resolve("default")
    assert binding.contract.get("api", "openai-chat") == api
    assert "fixture-key" not in json.dumps(binding.contract)
    explicit = command.create_parser().parse_args(arguments + ["--model-api", "anthropic-messages",
        "--model", "explicit-model", "--base-url", "https://explicit.invalid", "--api-key", "explicit-fixture"])
    assert _local_model_options(workspace, explicit) == {"api": "anthropic-messages", "model": "explicit-model",
        "base_url": "https://explicit.invalid", "api_key": "explicit-fixture"}
    assert _local_model_options(workspace, args) == options
    assert "fixture-key" not in (root / "config.yaml").read_text()


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_override", [False, True])
@pytest.mark.parametrize("api,model_name", [("openai-chat", "gpt-4o"), ("openai-responses", "gpt-5"),
                                            ("anthropic-messages", "claude-sonnet-4-5")])
async def test_cli_configuration_invokes_native_protocol_offline(monkeypatch, tmp_path, api, model_name, empty_override):
    from linktools.ai.model import _anthropic, _openai
    module = _anthropic if api == "anthropic-messages" else _openai
    requests = []

    def handle(request):
        requests.append(request)
        return httpx2.Response(200, json=response(api))

    monkeypatch.setattr(module, "DefaultAsyncHttpxClient", lambda **kwargs: httpx2.AsyncClient(
        transport=httpx2.MockTransport(handle), trust_env=False, **kwargs))
    prefix = "ANTHROPIC" if api == "anthropic-messages" else "OPENAI"
    base_url = "https://fixture.invalid" if api == "anthropic-messages" else "https://fixture.invalid/v1"
    monkeypatch.setenv(f"{prefix}_BASE_URL", base_url)
    monkeypatch.setenv(f"{prefix}_API_KEY", "fixture-key")
    args = run_command.create_parser().parse_args(["hello", "--model-api", api, "--model", model_name,
        "--base-url", "" if empty_override else base_url,
        "--api-key", "" if empty_override else "fixture-key"])
    options = _local_model_options(Workspace(tmp_path, {}), args)
    assert options["base_url"] == base_url
    assert options["api_key"] == "fixture-key"
    binding = _local_runtime_models(Workspace(tmp_path, {}), args).capture().resolve("default")
    async with binding.materialize() as model:
        result = await model.request([ModelRequest(parts=[UserPromptPart("hello")])], {"max_tokens": 128}, ModelRequestParameters())
    assert result.parts[0].content == "answer"
    request = requests[0]
    assert request.url.path == {"openai-chat": "/v1/chat/completions", "openai-responses": "/v1/responses",
                                "anthropic-messages": "/v1/messages"}[api]
    if api == "anthropic-messages":
        assert request.headers["x-api-key"] == "fixture-key"
        assert "authorization" not in request.headers
    else:
        assert request.headers["authorization"] == "Bearer fixture-key"


@pytest.mark.parametrize("provider,api", [("openai", "openai-chat"), ("openai", "openai-responses"),
                                         ("anthropic", "anthropic-messages")])
def test_cli_missing_provider_extra_has_install_hint(tmp_path, provider, api):
    script = """
import sys
sys.modules[sys.argv[1]] = None
from linktools.cli import CommandError
from linktools.commands.ai.run import command
args = command.create_parser().parse_args(["hello", "--project", sys.argv[3], "--model-api", sys.argv[2], "--model", "fixture"])
try:
    command.run(args)
except CommandError as error:
    assert "linktools-ai[" + sys.argv[1] + "]" in str(error), str(error)
else:
    raise AssertionError("missing extra did not fail")
"""
    result = subprocess.run([sys.executable, "-c", script, provider, api, str(tmp_path)],
                            env=dict(os.environ), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / ".linktools" / "runtime").exists()


def test_protocol_cache_selection_does_not_reuse_openai_connection(monkeypatch, tmp_path):
    config = run_command.environ.build_config("cli-protocol-cache", "LINKTOOLS_")
    monkeypatch.setattr(run_command.environ, "config", config)
    config.set("AI_MODEL_API", "anthropic-messages")
    config.set("OPENAI_MODEL", "wrong-model")
    config.set("OPENAI_BASE_URL", "https://wrong.invalid/v1")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    workspace = Workspace(tmp_path, {"model": " workspace-model "})
    args = run_command.create_parser().parse_args(["hello"])
    assert _local_model_options(workspace, args) == {"api": "anthropic-messages", "model": "workspace-model",
                                                 "base_url": None, "api_key": None}


@pytest.mark.parametrize("api", ["anthropic", "invalid", ["anthropic-messages"]])
def test_invalid_workspace_protocol_fails_before_model_assembly(tmp_path, api):
    from linktools.ai.errors import AIError, ErrorCode
    from argparse import Namespace
    with pytest.raises(AIError) as raised:
        _local_runtime_models(Workspace(tmp_path, {"model_api": api}),
                              Namespace(model_api=None, model="fixture", base_url=None, api_key=None))
    assert raised.value.code is ErrorCode.MODEL_CONFIG_INVALID
