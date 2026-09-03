#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenAI model binding."""

import math
from dataclasses import dataclass, field
from types import TracebackType
from urllib.parse import urlsplit, urlunsplit

import httpx2
from linktools.core import environ
from openai import AsyncOpenAI
from pydantic_ai import ModelSettings
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import DEFAULT_HTTP_TIMEOUT, Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.providers import Provider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.retries import AsyncHTTPX2TenacityTransport, RetryConfig, wait_retry_after
from tenacity import RetryCallState, retry_if_exception_type, stop_after_attempt, wait_fixed

from ..core import JsonValue, canonical_sha256
from ..errors import AIError, ErrorCode

_logger = environ.get_logger("ai.model.openai")


@dataclass(frozen=True, slots=True)
class _OpenAIModelBinding:
    route_id: str
    model: str
    base_url: "str | None" = None
    api_key: "str | None" = field(default=None, repr=False, compare=False)
    timeout: "float | None" = None
    max_retries: "int | None" = None
    retry_delay: "float | None" = None
    max_tokens: "int | None" = None
    context_window: "int | None" = None

    def __post_init__(self) -> None:
        model = self.model.strip().removeprefix("openai:")
        if not self.route_id.strip() or not model:
            raise ValueError("OpenAI model binding is incomplete")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "base_url", _normalize_base_url(self.base_url))
        if self.api_key is not None and not self.api_key.strip():
            object.__setattr__(self, "api_key", None)
        _validate_positive_number("timeout", self.timeout)
        _validate_non_negative_integer("max_retries", self.max_retries)
        _validate_non_negative_number("retry_delay", self.retry_delay)
        _validate_positive_integer("max_tokens", self.max_tokens)
        _validate_positive_integer("context_window", self.context_window)
        if self.retry_delay is not None and self.max_retries is None:
            raise ValueError("retry_delay requires max_retries")
        if self.max_tokens is not None and self.context_window is not None and self.max_tokens > self.context_window:
            raise ValueError("max_tokens cannot exceed context_window")

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
        if self.context_window is not None:
            settings["context_window"] = self.context_window
        return {
            "version": 1,
            "provider": self.provider,
            "model_identity": self.model_identity,
            "settings": settings,
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
                retry_delay=self.retry_delay,
            )
            settings: ModelSettings = {}
            if self.timeout is not None:
                settings["timeout"] = self.timeout
            if self.max_tokens is not None:
                settings["max_tokens"] = self.max_tokens
            profile: ModelProfile | None = None
            if self.context_window is not None:
                profile = {"context_window": self.context_window}
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


class _RetryingOpenAIProvider(Provider[AsyncOpenAI]):
    def __init__(
        self,
        *,
        base_url: "str | None",
        api_key: "str | None",
        max_retries: int,
        retry_delay: float,
    ) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._entered_count = 0
        self._http_client, self._delegate = self._build()

    @property
    def name(self) -> str:
        return self._delegate.name

    @property
    def base_url(self) -> str:
        return self._delegate.base_url

    @property
    def client(self) -> AsyncOpenAI:
        return self._delegate.client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        return OpenAIProvider.model_profile(model_name)

    async def __aenter__(self) -> "_RetryingOpenAIProvider":
        if self._entered_count == 0 and self._http_client.is_closed:
            self._http_client, self._delegate = self._build()
        self._entered_count += 1
        return self

    async def __aexit__(
        self,
        exc_type: "type[BaseException] | None",
        exc_value: "BaseException | None",
        traceback: "TracebackType | None",
    ) -> None:
        del exc_type, exc_value, traceback
        if self._entered_count == 0:
            return
        self._entered_count -= 1
        if self._entered_count == 0:
            await self._delegate.client.close()

    def _build(self) -> "tuple[httpx2.AsyncClient, OpenAIProvider]":
        http_client = _retry_http_client(
            max_retries=self._max_retries,
            retry_delay=self._retry_delay,
        )
        provider = OpenAIProvider(
            base_url=self._base_url,
            api_key=self._api_key,
            http_client=http_client,
        )
        provider.client.max_retries = 0
        return http_client, provider


class _AsyncClientTransport(httpx2.AsyncBaseTransport):
    def __init__(self, client: httpx2.AsyncClient) -> None:
        self._client = client

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        response = await self._client.send(request, stream=True)
        if _should_retry_response(response):
            await response.aread()
        return response

    async def aclose(self) -> None:
        await self._client.aclose()


def _openai_provider(
    *,
    base_url: "str | None",
    api_key: "str | None",
    max_retries: "int | None",
    retry_delay: "float | None",
) -> Provider[AsyncOpenAI]:
    if retry_delay is None:
        provider = OpenAIProvider(base_url=base_url, api_key=api_key)
        if max_retries is not None:
            provider.client.max_retries = max_retries
        return provider
    if max_retries is None:
        raise ValueError("retry_delay requires max_retries")
    return _RetryingOpenAIProvider(
        base_url=base_url,
        api_key=api_key,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )


def _retry_http_client(*, max_retries: int, retry_delay: float) -> httpx2.AsyncClient:
    timeout = httpx2.Timeout(timeout=DEFAULT_HTTP_TIMEOUT, connect=5)
    network_client = httpx2.AsyncClient(timeout=timeout)
    transport = AsyncHTTPX2TenacityTransport(
        config=RetryConfig(
            retry=retry_if_exception_type((httpx2.HTTPStatusError, httpx2.TransportError)),
            wait=wait_retry_after(fallback_strategy=wait_fixed(retry_delay)),
            stop=stop_after_attempt(max_retries + 1),
            retry_error_callback=_retry_error_result,
        ),
        wrapped=_AsyncClientTransport(network_client),
        validate_response=_raise_retryable_status,
    )
    return httpx2.AsyncClient(
        transport=transport,
        timeout=timeout,
    )


def _retry_error_result(state: RetryCallState) -> httpx2.Response:
    outcome = state.outcome
    if outcome is None or not outcome.failed:
        raise RuntimeError("retry exhausted without a failed attempt")
    error = outcome.exception()
    if isinstance(error, httpx2.HTTPStatusError):
        return error.response
    if error is None:
        raise RuntimeError("retry exhausted without an exception")
    raise error


def _raise_retryable_status(response: httpx2.Response) -> None:
    if _should_retry_response(response):
        response.raise_for_status()


def _should_retry_response(response: httpx2.Response) -> bool:
    if response.status_code < 400:
        return False
    should_retry = response.headers.get("x-should-retry")
    if should_retry == "false":
        return False
    if should_retry == "true":
        return True
    return response.status_code in {408, 409, 429} or response.status_code >= 500


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
