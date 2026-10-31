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
from linktools.commands.ai._common import _ModelOptions
from linktools.commands.ai.run import command as run_command
from linktools.commands.ai.acp import command as acp_command
from linktools.commands.ai.web import command as web_command
from tests.ai.test_model_protocols import response


@pytest.mark.parametrize("command,arguments", [(run_command, ["hello"]), (acp_command, []), (web_command, [])])
@pytest.mark.parametrize("api,prefix", [("openai", "OPENAI"), ("openai-responses", "OPENAI"),
                                       ("anthropic", "ANTHROPIC")])
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
    options = _ModelOptions.load(workspace, args)
    binding = options.create_model()
    assert binding.contract.get("api", "openai") == api
    assert binding.model_identity == binding.provider + f":{prefix.lower()}-model"
    assert binding.api_key == f"{prefix.lower()}-fixture-key"
    assert binding.base_url == f"https://{prefix.lower()}.invalid/v1"
    assert "fixture-key" not in json.dumps(binding.contract)
    assert "fixture-key" not in repr(options)
    explicit = command.create_parser().parse_args(arguments + ["--model-api", "anthropic",
        "--model", "explicit-model", "--base-url", "https://explicit.invalid", "--api-key", "explicit-fixture"])
    overridden = _ModelOptions.load(workspace, explicit).create_model()
    assert overridden.contract["api"] == "anthropic"
    assert overridden.model_identity == overridden.provider + ":explicit-model"
    assert overridden.base_url == "https://explicit.invalid"
    assert overridden.api_key == "explicit-fixture"
    assert _ModelOptions.load(workspace, args).create_model().contract == binding.contract
    assert "fixture-key" not in (root / "config.yaml").read_text()


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_override", [False, True])
@pytest.mark.parametrize("api,model_name", [("openai", "gpt-4o"), ("openai-responses", "gpt-5"),
                                            ("anthropic", "claude-sonnet-4-5")])
async def test_cli_configuration_invokes_native_protocol_offline(monkeypatch, tmp_path, api, model_name, empty_override):
    from linktools.ai.model import _anthropic, _openai
    module = _anthropic if api == "anthropic" else _openai
    requests = []

    def handle(request):
        requests.append(request)
        return httpx2.Response(200, json=response(api))

    monkeypatch.setattr(module, "DefaultAsyncHttpxClient", lambda **kwargs: httpx2.AsyncClient(
        transport=httpx2.MockTransport(handle), trust_env=False, **kwargs))
    prefix = "ANTHROPIC" if api == "anthropic" else "OPENAI"
    base_url = "https://fixture.invalid" if api == "anthropic" else "https://fixture.invalid/v1"
    monkeypatch.setenv(f"{prefix}_BASE_URL", base_url)
    monkeypatch.setenv(f"{prefix}_API_KEY", "fixture-key")
    args = run_command.create_parser().parse_args(["hello", "--model-api", api, "--model", model_name,
        "--base-url", "" if empty_override else base_url,
        "--api-key", "" if empty_override else "fixture-key"])
    binding = _ModelOptions.load(Workspace(tmp_path, {}), args).create_model()
    assert binding.base_url == base_url
    assert binding.api_key == "fixture-key"
    async with binding.materialize() as model:
        result = await model.request([ModelRequest(parts=[UserPromptPart("hello")])], {"max_tokens": 128}, ModelRequestParameters())
    assert result.parts[0].content == "answer"
    request = requests[0]
    assert request.url.path == {"openai": "/v1/chat/completions", "openai-responses": "/v1/responses",
                                "anthropic": "/v1/messages"}[api]
    if api == "anthropic":
        assert request.headers["x-api-key"] == "fixture-key"
        assert "authorization" not in request.headers
    else:
        assert request.headers["authorization"] == "Bearer fixture-key"


@pytest.mark.parametrize("provider,api", [("openai", "openai"), ("openai", "openai-responses"),
                                         ("anthropic", "anthropic")])
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
    config.set("MODEL_API", "anthropic")
    config.set("OPENAI_MODEL", "wrong-model")
    config.set("OPENAI_BASE_URL", "https://wrong.invalid/v1")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    workspace = Workspace(tmp_path, {"model": " workspace-model "})
    args = run_command.create_parser().parse_args(["hello"])
    binding = _ModelOptions.load(workspace, args).create_model()
    assert binding.contract["api"] == "anthropic"
    assert binding.model_identity == binding.provider + ":workspace-model"
    assert binding.base_url is None and binding.api_key is None


@pytest.mark.parametrize("api", ["anthropic-messages", "openai-chat", "invalid", ["anthropic"]])
def test_invalid_workspace_protocol_fails_before_model_assembly(tmp_path, api):
    from linktools.ai.errors import AIError, ErrorCode
    from argparse import Namespace
    with pytest.raises(AIError) as raised:
        _ModelOptions.load(Workspace(tmp_path, {"model_api": api}), Namespace(model_api=None, model="fixture", base_url=None, api_key=None)).create_model()
    assert raised.value.code is ErrorCode.MODEL_CONFIG_INVALID


@pytest.mark.parametrize("api,prefix", [("openai", "OPENAI"), ("openai-responses", "OPENAI"),
                                       ("anthropic", "ANTHROPIC")])
@pytest.mark.parametrize("source", ["profile", "environment", "native-environment", "cache"])
def test_cli_standard_config_sources_preserve_protocol_fields(tmp_path, api, prefix, source):
    home = tmp_path / "home"
    home.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    values = {"MODEL_API": api, f"{prefix}_MODEL": "source-model",
              f"{prefix}_BASE_URL": "https://source.invalid", f"{prefix}_API_KEY": "source-fixture-key"}
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("OPENAI_", "ANTHROPIC_", "LINKTOOLS_")) and key != "MODEL_API"}
    environment["HOME"] = str(home)
    if source == "profile":
        (home / ".linktools").mkdir()
        (home / ".linktools" / "linktools.json").write_text(json.dumps({"config": {
            "MODEL_API": "openai", f"{prefix}_MODEL": "global-model"}}))
        (project / ".linktools.json").write_text(json.dumps({"config": values}))
        # The existing profile source precedes prefixed process environment.
        environment[f"LINKTOOLS_{prefix}_MODEL"] = "environment-model"
    elif source in ("environment", "native-environment"):
        if source == "native-environment":
            environment[f"LINKTOOLS_{prefix}_MODEL"] = "prefixed-model"
        environment.update({("LINKTOOLS_" if source == "environment" else "") + key: value
                            for key, value in values.items()})
    else:
        environment[f"LINKTOOLS_{prefix}_API_KEY"] = values[f"{prefix}_API_KEY"]
    script = '''
import json, sys
from pathlib import Path
from linktools.core import environ
from linktools.ai.workspace import Workspace
from linktools.commands.ai._common import _ModelOptions
from linktools.commands.ai.run import command as run
from linktools.commands.ai.acp import command as acp
from linktools.commands.ai.web import command as web
api, prefix, source = sys.argv[1:]
environ.config.persist_many({"MODEL_API": api if source == "cache" else "openai",
    prefix + "_MODEL": "source-model" if source == "cache" else "cached-model",
    prefix + "_BASE_URL": "https://source.invalid" if source == "cache" else "https://cached.invalid"})
before = set(environ.config.persisted_keys())
workspace = Workspace(Path.cwd(), {"model": "workspace-model"})
for command, arguments in ((run, ["hello"]), (acp, []), (web, [])):
    args = command.create_parser().parse_args(arguments)
    binding = _ModelOptions.load(workspace, args).create_model()
    assert binding.contract.get("api", "openai") == api
    assert binding.model_identity == binding.provider + ":source-model"
    assert binding.base_url == "https://source.invalid"
    assert binding.api_key == "source-fixture-key"
    assert "source-fixture-key" not in json.dumps(binding.contract)
    explicit = command.create_parser().parse_args(arguments + ["--model-api", api, "--model", "cli-model",
        "--base-url", "https://cli.invalid", "--api-key", "cli-fixture-key"])
    overridden = _ModelOptions.load(workspace, explicit).create_model()
    assert overridden.model_identity == overridden.provider + ":cli-model"
    assert overridden.base_url == "https://cli.invalid"
    assert overridden.api_key == "cli-fixture-key"
    assert set(environ.config.persisted_keys()) == before
    again = _ModelOptions.load(workspace, args).create_model()
    assert again.api_key == "source-fixture-key"
    assert again.model_identity == binding.model_identity
assert not any("API_KEY" in key for key in environ.config.persisted_keys())
print("configuration sources verified")
'''
    result = subprocess.run([sys.executable, "-c", script, api, prefix, source],
                            cwd=project, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "fixture-key" not in result.stdout + result.stderr
    if source == "profile":
        assert json.loads((project / ".linktools.json").read_text()) == {"config": values}
