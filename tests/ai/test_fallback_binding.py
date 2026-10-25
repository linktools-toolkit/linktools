#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit fallback bindings retain native SDK policy and resource ownership."""

from collections.abc import Mapping
from types import TracebackType

import pytest
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

from linktools.ai.core import JsonValue
from linktools.ai.errors import AIError
from linktools.ai.model import (
    FallbackBinding, ModelBinding, create_model,
    validate_model_input,
)


class _Candidate(Model):
    def __init__(
        self, name: str, events: list[str], *, error: Exception | None = None,
        enter_error: Exception | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.events = events
        self.error = error
        self.enter_error = enter_error

    @property
    def model_name(self) -> str:
        return self.name

    @property
    def system(self) -> str:
        return "test"

    async def __aenter__(self) -> "_Candidate":
        self.events.append(f"enter:{self.name}")
        if self.enter_error is not None:
            raise self.enter_error
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.events.append(f"exit:{self.name}")

    async def request(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        self.events.append(f"request:{self.name}")
        if self.error is not None:
            raise self.error
        return ModelResponse([TextPart(self.name)], model_name=self.name)


class _Binding:
    provider = "test"

    def __init__(self, model: _Candidate) -> None:
        self.model = model

    @property
    def model_identity(self) -> str:
        return "test:" + self.model.name

    @property
    def contract(self) -> Mapping[str, JsonValue]:
        return {"model_identity": self.model_identity, "input_mime_types": []}

    def materialize(self) -> Model:
        self.model.events.append(f"materialize:{self.model.name}")
        return self.model


def test_fallback_requires_a_candidate_and_preserves_ordered_semantics() -> None:
    with pytest.raises(ValueError, match="at least one candidate"):
        FallbackBinding()
    primary = create_model(
        protocol="openai", model="primary", api_key="secret-primary",
        base_url="https://primary.example.test", max_tokens=25, input_types=["image"],
    )
    secondary = create_model(
        protocol="anthropic", model="secondary", api_key="secret-secondary",
        input_types=["mime:application/pdf"],
    )
    binding = FallbackBinding(primary, secondary)
    assert isinstance(binding, ModelBinding)
    assert binding.provider == "fallback"
    assert binding.contract["candidates"] == [
        {"provider": item.provider, "model_identity": item.model_identity, "contract": dict(item.contract)}
        for item in (primary, secondary)
    ]
    assert FallbackBinding(secondary, primary).contract != binding.contract
    assert FallbackBinding(secondary, primary).model_identity != binding.model_identity
    assert "secret" not in repr(binding.contract)
    assert "primary.example.test" not in repr(binding.contract)
    validate_model_input(binding, "image/png")
    validate_model_input(binding, "application/pdf")
    with pytest.raises(AIError):
        validate_model_input(binding, "audio/wav")


def test_materialize_prepares_ordered_candidates_without_mutating_shared_models() -> None:
    events: list[str] = []
    primary = _Binding(_Candidate("primary", events))
    secondary = _Binding(_Candidate("secondary", events))
    binding = FallbackBinding(primary, secondary)
    assert binding.contract["input_mime_types"] == []
    assert events == []
    prepared: list[ModelBinding] = []

    def prepare(candidate: ModelBinding) -> Model:
        prepared.append(candidate)
        return WrapperModel(candidate.materialize())

    first = binding.materialize(prepare=prepare)
    second = binding.materialize(prepare=prepare)
    assert isinstance(first, FallbackModel) and isinstance(second, FallbackModel)
    assert first is not second
    assert prepared == [primary, secondary, primary, secondary]
    assert [model.model_id for model in first.models] == [primary.model.model_id, secondary.model.model_id]
    assert all(left is not right for left, right in zip(first.models, second.models))
    direct = binding.materialize()
    assert [model.model_id for model in direct.models] == [primary.model.model_id, secondary.model.model_id]
    assert events == ["materialize:primary", "materialize:secondary"] * 3


def test_handler_implementation_and_connections_do_not_define_model_semantics() -> None:
    class Handler:
        def __call__(self, error: Exception) -> bool:
            return isinstance(error, ValueError)

        def __hash__(self) -> int:
            raise AssertionError("handler implementation must not be hashed")

        def __repr__(self) -> str:
            raise AssertionError("handler implementation must not be exposed")

    primary = create_model(protocol="openai", model="primary", api_key="first-secret")
    other_connection = create_model(
        protocol="openai", model="primary", api_key="second-secret",
        base_url="https://secondary.example.test", timeout=30, max_retries=5,
    )
    binding = FallbackBinding(primary, fallback_on=Handler())
    other = FallbackBinding(other_connection, fallback_on=ValueError)
    assert binding.contract == other.contract
    assert binding.model_identity == other.model_identity
    assert "secret" not in repr(binding)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_enter", (False, True))
async def test_native_fallback_owns_candidate_lifetimes(fail_enter: bool) -> None:
    events: list[str] = []
    error = RuntimeError("cannot enter secondary") if fail_enter else None
    primary = _Binding(_Candidate("primary", events))
    secondary = _Binding(_Candidate("secondary", events, enter_error=error))
    model = FallbackBinding(primary, secondary).materialize()

    async def enter_and_request() -> None:
        async with model:
            response = await model.request([], None, ModelRequestParameters())
            assert response.model_name == "primary"

    if fail_enter:
        with pytest.raises(RuntimeError) as raised:
            await enter_and_request()
        assert raised.value is error
        assert events[-3:] == ["enter:primary", "enter:secondary", "exit:primary"]
    else:
        await enter_and_request()
        assert events[-5:] == [
            "enter:primary", "enter:secondary", "request:primary", "exit:secondary", "exit:primary",
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("error_kind", (ModelAPIError, ValueError))
async def test_default_native_policy_falls_back_only_for_model_api_errors(
    error_kind: type[Exception],
) -> None:
    events: list[str] = []
    error = ModelAPIError("primary", "unavailable") if error_kind is ModelAPIError else ValueError("invalid")
    model = FallbackBinding(
        _Binding(_Candidate("primary", events, error=error)),
        _Binding(_Candidate("secondary", events)),
    ).materialize()
    if error_kind is ModelAPIError:
        response = await model.request([], None, ModelRequestParameters())
        assert response.model_name == "secondary"
        assert events[-2:] == ["request:primary", "request:secondary"]
    else:
        with pytest.raises(ValueError) as raised:
            await model.request([], None, ModelRequestParameters())
        assert raised.value is error
        assert events[-1] == "request:primary"


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ("type", "exception", "async-exception", "response", "async-response", "mixed"))
async def test_native_handler_types_and_response_annotations_are_preserved(policy: str) -> None:
    events: list[str] = []
    seen: list[Exception | ModelResponse] = []
    error = ValueError("handoff")

    def on_error(value: Exception) -> bool:
        seen.append(value)
        return value is error

    async def on_error_async(value: Exception) -> bool:
        return on_error(value)

    def on_response(value: ModelResponse) -> bool:
        seen.append(value)
        return value.model_name == "primary"

    async def on_response_async(value: ModelResponse) -> bool:
        return on_response(value)

    handlers = {
        "type": ValueError, "exception": on_error, "async-exception": on_error_async,
        "response": on_response, "async-response": on_response_async,
        "mixed": [ValueError, on_response],
    }
    binding = FallbackBinding(
        _Binding(_Candidate("primary", events, error=None if "response" in policy or policy == "mixed" else error)),
        _Binding(_Candidate("secondary", events)),
        fallback_on=handlers[policy],
    )
    response = await binding.materialize().request([], None, ModelRequestParameters())
    assert response.model_name == "secondary"
    assert events[-2:] == ["request:primary", "request:secondary"]
    if "response" in policy or policy == "mixed":
        assert [item.model_name for item in seen] == ["primary", "secondary"]
    elif policy != "type":
        assert seen == [error]


@pytest.mark.parametrize("policy", (ModelAPIError, ValueError))
def test_nested_binding_is_rejected_without_materializing_candidates(policy: type[Exception]) -> None:
    events: list[str] = []
    primary = _Binding(_Candidate("primary", events))
    secondary = _Binding(_Candidate("secondary", events))
    nested = FallbackBinding(primary, secondary)
    with pytest.raises(ValueError, match="nested fallback bindings are not supported"):
        FallbackBinding(nested, primary, fallback_on=policy)
    assert events == []


def test_three_candidates_prepare_in_declared_order() -> None:
    events: list[str] = []
    bindings = [_Binding(_Candidate(name, events)) for name in ("primary", "secondary", "third")]
    prepared: list[ModelBinding] = []

    def prepare(candidate: ModelBinding) -> Model:
        prepared.append(candidate)
        return WrapperModel(candidate.materialize())

    model = FallbackBinding(*bindings).materialize(prepare=prepare)
    assert prepared == bindings
    assert [item.model_name for item in model.models] == ["primary", "secondary", "third"]
    assert events == ["materialize:primary", "materialize:secondary", "materialize:third"]
