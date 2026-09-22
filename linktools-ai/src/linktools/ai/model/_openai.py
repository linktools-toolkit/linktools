#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenAI model binding."""

import asyncio
import math
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

from linktools.core import environ
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UserError
from pydantic_ai.messages import BinaryContent, ImageUrl, ModelRequest, UploadedFile
from pydantic_ai.models import (
    Model,
    ModelMessage,
    ModelRequestParameters,
    ModelResponse,
    ModelSettings,
    RunContext as PydanticRunContext,
    StreamedResponse,
)
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from ..core import JsonValue, canonical_sha256
from ..errors import AIError, ErrorCode

_logger = environ.get_logger("ai.model.openai")


@dataclass(frozen=True, slots=True)
class _OpenAIModelBinding:
    route_id: str
    model: str
    vision: bool = False
    base_url: "str | None" = None
    api_key: "str | None" = field(default=None, repr=False, compare=False)
    timeout: "int | float | None" = None
    max_retries: int = 2
    retry_delay: "int | float" = 1.0
    max_tokens: "int | None" = None
    connection_resolver: "Callable[[], Mapping[str, object]] | None" = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        model = self.model.strip().removeprefix("openai:")
        if not self.route_id.strip() or not model:
            raise ValueError("OpenAI model binding is incomplete")
        if type(self.vision) is not bool:
            raise ValueError("OpenAI model vision must be bool")
        if self.connection_resolver is not None and not callable(
            self.connection_resolver
        ):
            raise ValueError("OpenAI connection resolver must be callable")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "base_url", _normalize_base_url(self.base_url))
        if self.api_key is not None and not self.api_key.strip():
            object.__setattr__(self, "api_key", None)
        _validate_positive_number("timeout", self.timeout)
        _validate_non_negative_integer("max_retries", self.max_retries)
        _validate_non_negative_number("retry_delay", self.retry_delay)
        _validate_positive_integer("max_tokens", self.max_tokens)

    @property
    def provider(self) -> str:
        return "openai"

    @property
    def model_identity(self) -> str:
        return f"openai:{self.model}"

    @property
    def semantic_payload(self) -> "dict[str, JsonValue]":
        settings: dict[str, JsonValue] = {}
        if self.max_tokens is not None:
            settings["max_tokens"] = self.max_tokens
        return {
            "provider": self.provider,
            "model_identity": self.model_identity,
            "vision": self.vision,
            "settings": settings,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256({"contract": "model-v1", **self.semantic_payload})

    def materialize(self) -> Model:
        connection = _resolved_connection(self)
        try:
            provider = OpenAIProvider(
                base_url=connection["base_url"],
                api_key=connection["api_key"],
            )
            provider.client.max_retries = 0

            settings: ModelSettings = {}
            if connection["timeout"] is not None:
                settings["timeout"] = connection["timeout"]
            if self.max_tokens is not None:
                settings["max_tokens"] = self.max_tokens

            model: Model = OpenAIChatModel(
                self.model,
                provider=provider,
                settings=settings or None,
            )
        except UserError as error:
            raise AIError(
                ErrorCode.MODEL_CONFIG_INVALID,
                retryable=False,
                safe_details={
                    "provider": "openai",
                    "reason": "provider_configuration_invalid",
                },
            ) from error
        _logger.debug(
            "OpenAI model materialized: route=%s model=%s credential=%s",
            self.route_id,
            self.model,
            connection["api_key"] is not None,
        )
        return _RetryingModel(
            model,
            int(connection["max_retries"]),
            connection["retry_delay"],
            vision=self.vision,
        )


def _resolved_connection(binding: _OpenAIModelBinding) -> dict[str, object]:
    values: dict[str, object] = {
        "base_url": binding.base_url,
        "api_key": binding.api_key,
        "timeout": binding.timeout,
        "max_retries": binding.max_retries,
        "retry_delay": binding.retry_delay,
    }
    resolver = binding.connection_resolver
    if resolver is None:
        return values
    try:
        overrides = resolver()
    except AIError:
        raise
    except Exception as error:
        raise AIError(
            ErrorCode.MODEL_CONFIG_INVALID,
            safe_details={"reason": "connection_resolver_failed"},
        ) from error
    if not isinstance(overrides, Mapping):
        error = TypeError("connection resolver must return a mapping")
        raise AIError(
            ErrorCode.MODEL_CONFIG_INVALID,
            safe_details={"reason": "connection_resolver_invalid"},
        ) from error
    allowed = frozenset(values)
    if any(key not in allowed for key in overrides):
        error = ValueError("connection resolver returned an unsupported field")
        raise AIError(
            ErrorCode.MODEL_CONFIG_INVALID,
            safe_details={"reason": "connection_resolver_field_invalid"},
        ) from error
    values.update(overrides)
    try:
        if values["base_url"] is not None and not isinstance(
            values["base_url"], str
        ):
            raise ValueError("base_url must be a string")
        values["base_url"] = _normalize_base_url(values["base_url"])
        _validate_optional_string("api_key", values["api_key"])
        if isinstance(values["api_key"], str) and not values["api_key"].strip():
            values["api_key"] = None
        _validate_positive_number("timeout", values["timeout"])
        if values["max_retries"] is None:
            raise ValueError("max_retries must be a non-negative integer")
        _validate_non_negative_integer("max_retries", values["max_retries"])
        _validate_non_negative_number("retry_delay", values["retry_delay"])
    except (TypeError, ValueError) as error:
        raise AIError(
            ErrorCode.MODEL_CONFIG_INVALID,
            safe_details={"reason": "connection_resolver_value_invalid"},
        ) from error
    return values


def _validate_positive_number(name: str, value: "int | float | None") -> None:
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or (isinstance(value, float) and not math.isfinite(value))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number")


def _validate_optional_string(name: str, value: object) -> None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be a string")


def _validate_positive_integer(name: str, value: "int | None") -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_non_negative_integer(name: str, value: "int | None") -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_non_negative_number(name: str, value: "int | float") -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or (isinstance(value, float) and not math.isfinite(value))
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")


class _RetryingModel(WrapperModel):
    """Apply LinkTools' finite transport retry policy at the public Model boundary."""

    def __init__(
        self,
        wrapped: Model,
        max_retries: int,
        retry_delay: "int | float",
        *,
        vision: bool,
    ) -> None:
        super().__init__(wrapped)
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._vision = vision

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        _validate_vision_input(messages, vision=self._vision)
        attempt = 0
        while True:
            try:
                return await self.wrapped.request(
                    messages,
                    model_settings,
                    model_request_parameters,
                )
            except ModelAPIError as error:
                if attempt >= self._max_retries or not _retryable_model_error(error):
                    raise
                attempt += 1
                _logger.warning(
                    "retrying model request: model=%s attempt=%s",
                    self.model_name,
                    attempt,
                )
                await asyncio.sleep(self._retry_delay)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: "PydanticRunContext[object] | None" = None,
    ) -> AsyncGenerator[StreamedResponse, None]:
        _validate_vision_input(messages, vision=self._vision)
        attempt = 0
        while True:
            entered = False
            try:
                async with self.wrapped.request_stream(
                    messages,
                    model_settings,
                    model_request_parameters,
                    run_context,
                ) as response:
                    entered = True
                    yield response
                return
            except ModelAPIError as error:
                if (
                    entered
                    or attempt >= self._max_retries
                    or not _retryable_model_error(error)
                ):
                    raise
                attempt += 1
                _logger.warning(
                    "retrying streamed model request: model=%s attempt=%s",
                    self.model_name,
                    attempt,
                )
                await asyncio.sleep(self._retry_delay)


def _validate_vision_input(
    messages: Sequence[object],
    *,
    vision: bool,
) -> None:
    if vision:
        return
    if any(
        _contains_image_content(part.content)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if hasattr(part, "content")
    ):
        _logger.warning("OpenAI model rejected image input: vision=false")
        raise AIError(
            ErrorCode.REQUEST_FIELD_INVALID,
            retryable=False,
            safe_details={"reason": "image_input_not_supported"},
        )


def _contains_image_content(value: object) -> bool:
    if isinstance(value, BinaryContent):
        return value.media_type.lower().startswith("image/")
    if isinstance(value, ImageUrl):
        return True
    if isinstance(value, UploadedFile):
        return value.media_type.lower().startswith("image/")
    if isinstance(value, Mapping):
        return any(_contains_image_content(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return any(_contains_image_content(item) for item in value)
    return False


def _retryable_model_error(error: ModelAPIError) -> bool:
    if isinstance(error, ModelHTTPError):
        return error.status_code in {408, 409, 429} or 500 <= error.status_code < 600
    return True


def _normalize_base_url(value: "str | None") -> "str | None":
    if value is None or not value.strip():
        return None
    parsed = urlsplit(value.strip())
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("model base_url port is invalid") from error
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("model base_url must be a clean absolute HTTP URL")
    host = parsed.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    netloc = host if port is None else f"{host}:{port}"
    return urlunsplit(
        (parsed.scheme.lower(), netloc, parsed.path.rstrip("/"), "", "")
    )


__all__ = ["_OpenAIModelBinding"]
