#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

from linktools.ai.core.model_runtime import (
    ModelClientUnavailable,
    RuntimeModelConfig,
    build_model,
)


def _config(**overrides) -> RuntimeModelConfig:
    defaults = dict(
        model_type="standard",
        protocol="openai",
        model="gpt-4o-mini",
        base_url="https://api.example.com/v1",
        api_key="sk-test",
        auth_token=None,
        timeout_seconds=300,
        raw={"max_output_tokens": 4096, "max_turns": 10, "max_retries": 1},
    )
    defaults.update(overrides)
    return RuntimeModelConfig(**defaults)


def test_build_model_takes_a_runtime_model_config_directly():
    bundle = build_model(_config())
    assert bundle.config.model_type == "standard"
    assert bundle.model.model_name == "gpt-4o-mini"
    assert bundle.settings["max_tokens"] == 4096
    assert bundle.usage_limits.request_limit == 10


def test_build_model_rejects_unsupported_protocol():
    with pytest.raises(ModelClientUnavailable, match="unsupported protocol"):
        build_model(_config(protocol="anthropic"))


def test_build_model_rejects_missing_base_url():
    with pytest.raises(ModelClientUnavailable, match="requires base_url"):
        build_model(_config(base_url=""))
