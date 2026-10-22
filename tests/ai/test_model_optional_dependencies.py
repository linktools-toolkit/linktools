#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run this module in each installed wheel extra combination without source PYTHONPATH."""
from ._runtime_test_helpers import model_features
import importlib.util
import os
from dataclasses import dataclass

import pytest
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.models.test import TestModel

from linktools.ai.errors import ErrorCode
from linktools.ai.model import create_model, model_binding_error
from linktools.ai.runtime import Runtime, RuntimeStorage
import linktools.ai.web


def test_installed_provider_set_and_missing_extra_hints():
    expected = os.getenv("EXPECTED_PROVIDER_EXTRAS")
    if expected is not None:
        assert {name for name in ("openai", "anthropic") if importlib.util.find_spec(name)} == set(filter(None, expected.split(",")))
    for provider, factories in (("openai", ("openai", "openai-responses")), ("anthropic", ("anthropic",))):
        for factory in factories:
            if importlib.util.find_spec(provider) is None:
                with pytest.raises(ImportError, match=rf"linktools-ai\[{provider}\]"):
                    create_model(protocol=factory, model="fixture")
            else:
                assert create_model(protocol=factory, model="fixture")
    assert model_binding_error(ModelHTTPError(429, model_name="fixture")).code == ErrorCode.MODEL_RATE_LIMITED
    assert model_binding_error(ValueError("ordinary failure")) is None


@dataclass(frozen=True)
class _TestBinding:
    route_id: str = "default"
    provider: str = "fixture"
    model_identity: str = "fixture:test"
    @property
    def contract(self): return {"provider": self.provider, "input_mime_types": []}
    def materialize(self): return TestModel(custom_output_text="base runtime works")


@pytest.mark.asyncio
async def test_base_runtime_can_invoke_custom_sdk_model():
    models = _TestBinding()
    async with Runtime.open("extras", features=model_features(models), storage=RuntimeStorage.in_memory()) as runtime:
        execution = await runtime.agents.get().start("hello")
        result = await execution.wait()
        assert result.result.status.value == "SUCCEEDED"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider,factory", (("openai","openai"),("openai","openai-responses"),("anthropic","anthropic")))
async def test_installed_adapter_projects_native_transport_errors(monkeypatch, provider, factory):
    if importlib.util.find_spec(provider) is None:
        pytest.skip("Provider intentionally absent from this installation")
    import httpx2
    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from pydantic_ai.models import ModelRequestParameters
    module = importlib.import_module(f"linktools.ai.model._{provider}")
    def fail(request): raise httpx2.ConnectError("fixture connection failure", request=request)
    monkeypatch.setattr(module, "DefaultAsyncHttpxClient", lambda **kwargs: httpx2.AsyncClient(
        transport=httpx2.MockTransport(fail), trust_env=False, **kwargs))
    models = create_model(protocol=factory, model="fixture", api_key="fixture-secret", max_tokens=128, max_retries=0)
    async with models.materialize() as model:
        with pytest.raises(ModelAPIError) as caught:
            await model.request([ModelRequest(parts=[UserPromptPart("hello")])], None, ModelRequestParameters())
    mapped = model_binding_error(caught.value)
    assert mapped is not None and mapped.code == ErrorCode.MODEL_API_ERROR
    sdk = importlib.import_module(provider)
    native = sdk.APIConnectionError(request=httpx2.Request("POST", "https://model.invalid"))
    assert model_binding_error(native).code == ErrorCode.MODEL_UNAVAILABLE
