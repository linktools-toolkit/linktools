#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenAI model binding."""

import math
from dataclasses import dataclass, field
from typing import cast
from urllib.parse import urlsplit, urlunsplit

import httpx2
from linktools.core import environ
from pydantic_ai import ModelSettings
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.retries import AsyncHTTPX2TenacityTransport, RetryConfig, wait_retry_after
from tenacity import retry_if_exception_type, stop_after_attempt, wait_fixed

from ..core import JsonValue, canonical_sha256
from ..errors import AIError, ErrorCode

_logger = environ.get_logger("ai.model.openai")


@dataclass(frozen=True, slots=True)
class _OpenAIModelBinding:
    route_id: str
    model: str
    base_url: "str | None" = None
    api_key: "str | None" = field(default=None, repr=False, compare=False)
    timeout_seconds: "float | None" = None
    max_retries: "int | None" = None
    retry_delay_seconds: "float | None" = None
    max_output_tokens: "int | None" = None
    context_window_tokens: "int | None" = None

    def __post_init__(self) -> None:
        model = self.model.strip().removeprefix("openai:")
        if not self.route_id.strip() or not model:
            raise ValueError("OpenAI model binding is incomplete")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "base_url", _normalize_base_url(self.base_url))
        if self.api_key is not None and not self.api_key.strip():
            object.__setattr__(self, "api_key", None)
        _validate_positive_number("timeout_seconds", self.timeout_seconds)
        _validate_non_negative_integer("max_retries", self.max_retries)
        _validate_non_negative_number("retry_delay_seconds", self.retry_delay_seconds)
        _validate_positive_integer("max_output_tokens", self.max_output_tokens)
        _validate_positive_integer("context_window_tokens", self.context_window_tokens)
        if self.retry_delay_seconds is not None and self.max_retries is None:
            raise ValueError("retry_delay_seconds requires max_retries")
        if (
            self.max_output_tokens is not None
            and self.context_window_tokens is not None
            and self.max_output_tokens > self.context_window_tokens
        ):
            raise ValueError("max_output_tokens cannot exceed context_window_tokens")

    @property
    def provider(self) -> str:
        return "openai"

    @property
    def model_identity(self) -> str:
        return f"openai:{self.model}"

    @property
    def semantic_payload(self) -> "dict[str, JsonValue]":
        return {
            "version": 1,
            "provider": self.provider,
            "model_identity": self.model_identity,
            "settings": {},
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256({"contract": "model-v1", **self.semantic_payload})

    def materialize(self) -> Model:
        try:
            provider = _openai_provider(
                base_url=self.base_url,
                api_key=self.api_key,
                max_retries=self.max_retries,
                retry_delay_seconds=self.retry_delay_seconds,
            )
            settings: ModelSettings = {}
            if self.timeout_seconds is not None:
                settings["timeout"] = self.timeout_seconds
            if self.max_output_tokens is not None:
                settings["max_tokens"] = self.max_output_tokens
            profile = None
            if self.context_window_tokens is not None:
                profile = cast(ModelProfile, {"context_window": self.context_window_tokens})
            model = OpenAIChatModel(
                self.model,
                provider=provider,
                settings=settings or None,
                profile=profile,
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
            self.api_key is not None,
        )
        return model


def _openai_provider(
    *,
    base_url: "str | None",
    api_key: "str | None",
    max_retries: "int | None",
    retry_delay_seconds: "float | None",
) -> OpenAIProvider:
    if retry_delay_seconds is None:
        provider = OpenAIProvider(base_url=base_url, api_key=api_key)
        if max_retries is not None:
            provider.client.max_retries = max_retries
        return provider

    transport = AsyncHTTPX2TenacityTransport(
        config=RetryConfig(
            retry=retry_if_exception_type((httpx2.HTTPStatusError, httpx2.TransportError)),
            wait=wait_retry_after(fallback_strategy=wait_fixed(retry_delay_seconds)),
            stop=stop_after_attempt(cast(int, max_retries) + 1),
            reraise=True,
        ),
        validate_response=_raise_retryable_status,
    )
    provider = OpenAIProvider(
        base_url=base_url,
        api_key=api_key,
        http_client=httpx2.AsyncClient(transport=transport),
    )
    provider.client.max_retries = 0
    return provider


def _raise_retryable_status(response: httpx2.Response) -> None:
    status_code = response.status_code
    if status_code in {408, 409, 429} or status_code >= 500:
        response.raise_for_status()


def _validate_positive_number(name: str, value: "float | None") -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")


def _validate_non_negative_number(name: str, value: "float | None") -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")


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


def _normalize_base_url(value: "str | None") -> "str | None":
    if value is None or not value.strip():
        return None
    parsed = urlsplit(value.strip())
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("model base_url port is invalid") from error
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        raise ValueError("model base_url must be a clean absolute HTTP URL")
    host = parsed.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    return urlunsplit((parsed.scheme.lower(), host if port is None else f"{host}:{port}", parsed.path.rstrip("/"), "", ""))


__all__ = ["_OpenAIModelBinding"]
