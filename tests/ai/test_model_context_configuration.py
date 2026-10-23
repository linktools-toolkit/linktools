#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic model limits share one API/config vocabulary and effective profile."""

import pytest
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters

from linktools.ai.core import PromptLimits
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import FallbackBinding, create_model
from linktools.ai.runtime._compaction import CompactionCapability
from linktools.ai.workspace import Workspace
from linktools.commands.ai._common import _local_model, _local_runtime_model
from linktools.commands.ai.run import command as run_command
from linktools.commands.ai.acp import command as acp_command
from linktools.commands.ai.web import command as web_command


@pytest.mark.parametrize("protocol", ("openai", "openai-responses", "anthropic"))
@pytest.mark.parametrize("explicit", (None, 131072))
def test_context_window_uses_one_effective_profile_without_mutating_input(protocol, explicit):
    profile = {"context_window": 65536, "supports_tools": True}
    binding = create_model(protocol=protocol, model="synthetic", api_key="synthetic",
                           profile=profile, context_window=explicit, max_tokens=8192)
    expected = 65536 if explicit is None else explicit
    model = binding.materialize()
    assert model.context_window == expected
    assert model.settings["max_tokens"] == 8192
    assert binding.contract["profile"] == {"context_window": expected, "supports_tools": True}
    assert "context_window" not in binding.contract
    assert profile == {"context_window": 65536, "supports_tools": True}
    request = ModelRequestContext(model=model, messages=[], model_settings=None,
                                  model_request_parameters=ModelRequestParameters())
    assert CompactionCapability(None, limits=PromptLimits())._request_target(request) == expected - 8192


@pytest.mark.parametrize("value", (True, False, 0, -1, 1.5, "131072"))
def test_explicit_context_window_rejects_non_positive_integer_tokens(value):
    with pytest.raises(ValueError, match="context_window"):
        create_model(protocol="openai", model="synthetic", context_window=value)


def test_context_window_defaults_and_fallback_use_native_profile_resolution():
    default = create_model(protocol="openai", model="synthetic", api_key="synthetic")
    assert "profile" not in default.contract
    assert default.materialize().context_window is None
    candidates = [create_model(protocol="openai", model="synthetic", api_key="synthetic", context_window=size)
                  for size in (131072, 32768)]
    assert FallbackBinding(*candidates, default).materialize().context_window == 32768
    assert FallbackBinding(default).materialize().context_window is None


@pytest.mark.parametrize(("command", "arguments"), ((run_command, ["hello"]), (acp_command, []), (web_command, [])))
def test_cli_numeric_options_override_config_environment_and_workspace(monkeypatch, tmp_path, command, arguments):
    config = command.environ.build_config("model-generic-options", "LINKTOOLS_")
    monkeypatch.setattr(command.environ, "config", config)
    values = {"context_window": 131072, "max_tokens": 8192, "timeout": 45,
              "max_retries": 3, "retry_delay": 5, "retry_max_delay": 120, "retry_backoff_factor": 4}
    workspace = Workspace(tmp_path, {"model": "synthetic", **values})
    for name in values:
        monkeypatch.delenv("LINKTOOLS_MODEL_" + name.upper(), raising=False)
    monkeypatch.setenv("LINKTOOLS_MODEL_CONTEXT_WINDOW", "65536")
    args = command.create_parser().parse_args(arguments + ["--model", "synthetic", "--api-key", "synthetic"])
    binding = _local_runtime_model(workspace, args)
    assert binding.contract["profile"]["context_window"] == 65536
    assert binding.max_tokens == 8192 and binding.timeout == 45
    assert binding.max_retries == 3 and binding.retry_delay == 5
    assert binding.retry_max_delay == 120 and binding.retry_backoff_factor == 4
    options = []
    for name, value in values.items():
        options.extend(("--" + name.replace("_", "-"), str(value)))
    args = command.create_parser().parse_args(arguments + ["--model", "synthetic", "--api-key", "synthetic", *options])
    assert _local_runtime_model(workspace, args).contract["profile"]["context_window"] == 131072


@pytest.mark.parametrize(("name", "value"), (("context_window", True), ("max_tokens", 0), ("timeout", -1),
    ("max_retries", -1), ("retry_delay", -1), ("retry_max_delay", float("inf")), ("retry_backoff_factor", .5)))
def test_workspace_numeric_limits_use_existing_constructor_validation(monkeypatch, tmp_path, name, value):
    monkeypatch.setattr(run_command.environ, "config", run_command.environ.build_config("invalid-model-limits", "LINKTOOLS_"))
    for option in ("context_window", "max_tokens", "timeout", "max_retries", "retry_delay", "retry_max_delay", "retry_backoff_factor"):
        monkeypatch.delenv("LINKTOOLS_MODEL_" + option.upper(), raising=False)
    with pytest.raises(AIError) as failed:
        _local_model(Workspace(tmp_path, {"model": "synthetic", name: value}), api_key="synthetic")
    assert failed.value.code is ErrorCode.MODEL_CONFIG_INVALID


def test_missing_generic_configuration_keeps_library_defaults(monkeypatch, tmp_path):
    monkeypatch.setattr(run_command.environ, "config", run_command.environ.build_config("default-model-limits", "LINKTOOLS_"))
    for option in ("context_window", "max_tokens", "timeout", "max_retries", "retry_delay", "retry_max_delay", "retry_backoff_factor"):
        monkeypatch.delenv("LINKTOOLS_MODEL_" + option.upper(), raising=False)
    binding = _local_model(Workspace(tmp_path, {"model": "synthetic"}), api_key="synthetic")
    assert "profile" not in binding.contract
    assert binding.max_tokens is None and binding.timeout is None
    assert (binding.max_retries, binding.retry_delay, binding.retry_max_delay, binding.retry_backoff_factor) == (2, 1, 60, 2)


def test_explicit_cli_zero_is_not_replaced_by_configured_retry_values(monkeypatch, tmp_path):
    monkeypatch.setattr(run_command.environ, "config", run_command.environ.build_config("zero-model-limits", "LINKTOOLS_"))
    monkeypatch.setenv("LINKTOOLS_MODEL_MAX_RETRIES", "3")
    monkeypatch.setenv("LINKTOOLS_MODEL_RETRY_DELAY", "5")
    args = run_command.create_parser().parse_args([
        "hello", "--model", "synthetic", "--api-key", "synthetic",
        "--max-retries", "0", "--retry-delay", "0", "--retry-max-delay", "0", "--retry-backoff-factor", "1",
    ])
    binding = _local_runtime_model(Workspace(tmp_path, {"max_retries": 2, "retry_delay": 10}), args)
    assert (binding.max_retries, binding.retry_delay, binding.retry_max_delay, binding.retry_backoff_factor) == (0, 0, 0, 1)


def test_persisted_model_limit_is_read_before_workspace_without_copying_it(monkeypatch, tmp_path):
    import os
    from linktools.core import Config, ConfigSchema, ConfigStore, DefaultSource, EnvironmentSource, PersistentSource

    store = ConfigStore(tmp_path / "settings.json")
    persistent = PersistentSource(store, "model-test")
    persistent.set("MODEL_CONTEXT_WINDOW", 65536)
    schema = ConfigSchema()
    config = Config(None, schema, sources=[EnvironmentSource((os.environ, "LINKTOOLS_")), persistent, DefaultSource(schema)])
    monkeypatch.setattr(run_command.environ, "config", config)
    monkeypatch.delenv("LINKTOOLS_MODEL_CONTEXT_WINDOW", raising=False)
    workspace = Workspace(tmp_path, {"model": "synthetic", "context_window": 32768})
    binding = _local_model(workspace, api_key="synthetic")
    assert binding.contract["profile"]["context_window"] == 65536
    monkeypatch.setenv("LINKTOOLS_MODEL_CONTEXT_WINDOW", "131072")
    assert _local_model(workspace, api_key="synthetic").contract["profile"]["context_window"] == 131072
    assert persistent.get("MODEL_CONTEXT_WINDOW") == (65536, True)
    assert workspace.config["context_window"] == 32768


@pytest.mark.parametrize(("name", "value"), (("TIMEOUT", "nan"), ("RETRY_DELAY", "inf"),
    ("RETRY_MAX_DELAY", "nan"), ("RETRY_BACKOFF_FACTOR", "inf"), ("CONTEXT_WINDOW", "1.5")))
def test_invalid_numeric_environment_does_not_silently_fall_back(monkeypatch, tmp_path, name, value):
    monkeypatch.setattr(run_command.environ, "config", run_command.environ.build_config("bad-model-env", "LINKTOOLS_"))
    monkeypatch.setenv("LINKTOOLS_MODEL_" + name, value)
    with pytest.raises(AIError) as failed:
        _local_model(Workspace(tmp_path, {"model": "synthetic"}), api_key="synthetic")
    assert failed.value.code is ErrorCode.MODEL_CONFIG_INVALID
