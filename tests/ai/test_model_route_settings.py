#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model route configuration coverage."""

from typing import Any

import pytest
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelBinding, ModelResolver, create_model
from linktools.ai.model._binding import _RetryingModel
from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    UploadedFile,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters, ModelResponse, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.openai import OpenAIProvider


class _CountingModel(TestModel):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        self.calls += 1
        return await super().request(
            messages,
            model_settings,
            model_request_parameters,
        )


def test_openai_operational_settings_do_not_change_model_contract() -> None:
    first = create_model(protocol="openai",
        model="gpt-test",
        base_url="https://first.example/v1",
        api_key="first-key",
        timeout=30,
        max_retries=1,
        max_tokens=2048,
    )
    second = create_model(protocol="openai",
        model="openai:gpt-test",
        base_url="https://second.example/v1",
        api_key="second-key",
        timeout=60,
        max_retries=3,
        max_tokens=2048,
    )

    assert dict(first.contract) == dict(second.contract)


def test_openai_custom_endpoint_is_operational_configuration() -> None:
    binding = create_model(protocol="openai",
        model="gpt-test",
        base_url="https://gateway.example/v1",
    )

    assert dict(binding.contract) == {
        "provider": "openai",
        "model_identity": "openai:gpt-test",
        "input_mime_types": [],
        "settings": {},
    }


def _input_binding(*, allow_files: bool = False) -> ModelBinding:
    return create_model(protocol="openai",
        model="gpt-test", input_types=("mime:image/*", "mime:application/*", "mime:text/*") if allow_files else (),
    )


def test_openai_input_policy_is_durable_model_contract() -> None:
    assert _input_binding().contract != _input_binding(allow_files=True).contract


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    (
        BinaryContent(b"image", media_type="image/png"),
        ImageUrl("https://example.com/image.png"),
        UploadedFile("report.png", "openai"),
    ),
)
async def test_openai_with_reject_policy_rejects_images_before_provider(
    content: Any,
) -> None:
    wrapped = _CountingModel()
    model = _RetryingModel(wrapped, 2, 0, input_binding=_input_binding())

    with pytest.raises(AIError) as raised:
        await model.request(
            [ModelRequest(parts=[UserPromptPart([content])])],
            None,
            ModelRequestParameters(),
        )

    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID
    assert raised.value.retryable is False
    assert raised.value.safe_details["reason"] == "model_input_not_supported"
    assert wrapped.calls == 0


@pytest.mark.asyncio
async def test_openai_with_reject_policy_rejects_stream_images_before_provider() -> None:
    wrapped = _CountingModel()
    model = _RetryingModel(wrapped, 2, 0, input_binding=_input_binding())

    with pytest.raises(AIError) as raised:
        async with model.request_stream(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            [BinaryContent(b"image", media_type="image/png")]
                        )
                    ]
                )
            ],
            None,
            ModelRequestParameters(),
        ):
            raise AssertionError("input policy should reject before stream entry")

    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID
    assert raised.value.retryable is False
    assert wrapped.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("media_type", ("application/pdf", "text/plain"))
async def test_openai_allow_files_allows_non_image_attachments(
    media_type: str,
) -> None:
    wrapped = _CountingModel()
    model = _RetryingModel(wrapped, 2, 0, input_binding=_input_binding(allow_files=True))

    await model.request(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        [BinaryContent(b"document", media_type=media_type)]
                    )
                ]
            )
        ],
        None,
        ModelRequestParameters(),
    )

    assert wrapped.calls == 1


@pytest.mark.asyncio
async def test_openai_input_policy_is_independent_per_binding() -> None:
    parent_provider = _CountingModel()
    child_provider = _CountingModel()
    parent_model = _RetryingModel(parent_provider, 2, 0, input_binding=_input_binding())
    child_model = _RetryingModel(child_provider, 2, 0, input_binding=_input_binding(allow_files=True))
    messages = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    [BinaryContent(b"image", media_type="image/png")]
                )
            ]
        )
    ]

    with pytest.raises(AIError):
        await parent_model.request(messages, None, ModelRequestParameters())
    await child_model.request(messages, None, ModelRequestParameters())

    assert parent_provider.calls == 0
    assert child_provider.calls == 1


@pytest.mark.asyncio
async def test_openai_allow_files_does_not_guess_opaque_uploaded_file_type() -> None:
    wrapped = _CountingModel()
    model = _RetryingModel(wrapped, 2, 0, input_binding=_input_binding(allow_files=True))

    uploaded = UploadedFile("file-image", "openai")
    assert uploaded.media_type == "application/octet-stream"

    await model.request(
        [ModelRequest(parts=[UserPromptPart([uploaded])])],
        None,
        ModelRequestParameters(),
    )

    assert wrapped.calls == 1


def test_openai_max_tokens_changes_model_contract() -> None:
    plain = create_model(protocol="openai", model="gpt-test")
    configured = create_model(protocol="openai",
        model="gpt-test",
        max_tokens=2048,
    )

    assert dict(plain.contract)["settings"] == {}
    assert dict(configured.contract)["settings"] == {"max_tokens": 2048}
    assert dict(plain.contract) != dict(configured.contract)


def test_model_registry_restore_requires_exact_model_contract() -> None:
    historical = create_model(protocol="openai",
        model="gpt-test",
        max_tokens=1024,
    )
    registry = create_model(protocol="openai",
        model="gpt-test",
        max_tokens=2048,
    )

    with pytest.raises(AIError) as raised:
        ModelResolver({"default": registry}).restore(
            dict(historical.contract),
            route_id="default",
        )

    assert raised.value.code is ErrorCode.AGENT_BINDING_UNAVAILABLE


def test_openai_route_materializes_settings_and_retries() -> None:
    binding = create_model(protocol="openai",
        model="gpt-test",
        api_key="test-key",
        timeout=30,
        max_retries=1,
        max_tokens=2048,
    )

    model = binding.materialize()

    assert isinstance(model, _RetryingModel)
    assert isinstance(model.wrapped, OpenAIChatModel)
    assert model.wrapped.settings == {"timeout": 30, "max_tokens": 2048}
    provider = model.wrapped.provider
    assert isinstance(provider, OpenAIProvider)
    assert provider.client.max_retries == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout": 0},
        {"timeout": float("inf")},
        {"timeout": True},
        {"max_retries": -1},
        {"max_retries": True},
        {"max_tokens": 0},
        {"max_tokens": True},
    ],
)
def test_openai_route_rejects_invalid_settings(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        create_model(protocol="openai", model="gpt-test", **kwargs)
