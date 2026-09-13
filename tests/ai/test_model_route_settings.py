#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model route configuration coverage."""

from typing import Any

import pytest
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.model._openai import _RetryingModel
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


def test_openai_operational_settings_do_not_change_durable_identity() -> None:
    first = ModelRegistry.openai(
        model="gpt-test",
        base_url="https://first.example/v1",
        api_key="first-key",
        timeout=30,
        max_retries=1,
        max_tokens=2048,
    ).snapshot().resolve("default")
    second = ModelRegistry.openai(
        model="openai:gpt-test",
        base_url="https://second.example/v1",
        api_key="second-key",
        timeout=60,
        max_retries=3,
        max_tokens=2048,
    ).snapshot().resolve("default")

    assert dict(first.semantic_payload) == dict(second.semantic_payload)
    assert first.fingerprint == second.fingerprint


def test_openai_custom_endpoint_is_operational_configuration() -> None:
    binding = ModelRegistry.openai(
        model="gpt-test",
        base_url="https://gateway.example/v1",
    ).snapshot().resolve("default")

    assert dict(binding.semantic_payload) == {
        "provider": "openai",
        "model_identity": "openai:gpt-test",
        "vision": False,
        "settings": {},
    }


def test_openai_vision_is_durable_model_semantics() -> None:
    without_vision = ModelRegistry.openai(
        model="gpt-test",
        vision=False,
    ).snapshot().resolve("default")
    with_vision = ModelRegistry.openai(
        model="gpt-test",
        vision=True,
    ).snapshot().resolve("default")

    assert dict(without_vision.semantic_payload)["vision"] is False
    assert dict(with_vision.semantic_payload)["vision"] is True
    assert without_vision.fingerprint != with_vision.fingerprint


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    (
        BinaryContent(b"image", media_type="image/png"),
        ImageUrl("https://example.com/image"),
        UploadedFile("report.png", "openai"),
    ),
)
async def test_openai_without_vision_rejects_images_before_provider(
    content: Any,
) -> None:
    wrapped = _CountingModel()
    model = _RetryingModel(wrapped, 2, 0, vision=False)

    with pytest.raises(AIError) as raised:
        await model.request(
            [ModelRequest(parts=[UserPromptPart([content])])],
            None,
            ModelRequestParameters(),
        )

    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID
    assert raised.value.retryable is False
    assert raised.value.safe_details == {
        "reason": "image_input_not_supported",
    }
    assert wrapped.calls == 0


@pytest.mark.asyncio
async def test_openai_without_vision_rejects_stream_images_before_provider() -> None:
    wrapped = _CountingModel()
    model = _RetryingModel(wrapped, 2, 0, vision=False)

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
            raise AssertionError("vision guard should reject before stream entry")

    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID
    assert raised.value.retryable is False
    assert wrapped.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("media_type", ("application/pdf", "text/plain"))
async def test_openai_without_vision_allows_non_image_attachments(
    media_type: str,
) -> None:
    wrapped = _CountingModel()
    model = _RetryingModel(wrapped, 2, 0, vision=False)

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
async def test_openai_vision_policy_is_independent_per_binding() -> None:
    parent_provider = _CountingModel()
    child_provider = _CountingModel()
    parent_model = _RetryingModel(parent_provider, 2, 0, vision=False)
    child_model = _RetryingModel(child_provider, 2, 0, vision=True)
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
async def test_openai_without_vision_does_not_guess_opaque_uploaded_file_type() -> None:
    wrapped = _CountingModel()
    model = _RetryingModel(wrapped, 2, 0, vision=False)

    uploaded = UploadedFile("file-image", "openai")
    assert uploaded.media_type == "application/octet-stream"

    await model.request(
        [ModelRequest(parts=[UserPromptPart([uploaded])])],
        None,
        ModelRequestParameters(),
    )

    assert wrapped.calls == 1


def test_openai_max_tokens_changes_durable_identity() -> None:
    plain = ModelRegistry.openai(model="gpt-test").snapshot().resolve("default")
    configured = ModelRegistry.openai(
        model="gpt-test",
        max_tokens=2048,
    ).snapshot().resolve("default")

    assert dict(plain.semantic_payload)["settings"] == {}
    assert dict(configured.semantic_payload)["settings"] == {"max_tokens": 2048}
    assert plain.fingerprint != configured.fingerprint


def test_model_registry_restore_requires_exact_semantic_settings() -> None:
    historical = ModelRegistry.openai(
        model="gpt-test",
        max_tokens=1024,
    ).snapshot().resolve("default")
    registry = ModelRegistry.openai(
        model="gpt-test",
        max_tokens=2048,
    )

    with pytest.raises(AIError) as raised:
        registry.snapshot().restore(
            dict(historical.semantic_payload),
            route_id="default",
        )

    assert raised.value.code is ErrorCode.AGENT_DEFINITION_UNAVAILABLE


def test_openai_route_materializes_settings_and_retries() -> None:
    binding = ModelRegistry.openai(
        model="gpt-test",
        api_key="test-key",
        timeout=30,
        max_retries=1,
        max_tokens=2048,
    ).snapshot().resolve("default")

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
        ModelRegistry.openai(model="gpt-test", **kwargs)
