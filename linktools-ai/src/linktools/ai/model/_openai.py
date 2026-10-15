#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenAI model binding."""

import asyncio
import math
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from types import TracebackType
from urllib.parse import urlsplit, urlunsplit

from httpx2 import AsyncByteStream, ByteStream, DecodingError, EventSource, Limits, Request, RequestNotRead, Response, SSEError
from linktools.core import environ
from openai import DefaultAsyncHttpxClient
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UserError
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

from ..core import JsonValue
from ..errors import AIError, ErrorCode
from ._client_request import ModelClientResponse, _ClientRequestEntry, _record_client_request
from ._contract import ModelBinding
from ._input import ModelInputCapability, UnknownModelInputs, normalize_input_capabilities, validate_model_inputs
from ._openai_input import openai_input_transport

_logger = environ.get_logger("ai.model.openai")
_CLIENT_REQUEST_ENTRY = "linktools.model.client_request_entry"


async def _capture_chat_request(request: Request) -> None:
    if request.method != "POST" or not request.url.path.endswith("/chat/completions"):
        return
    try:
        body = request.content
    except RequestNotRead:
        entry = _record_client_request("openai-chat", None, capture_error="body_unavailable")
    else:
        entry = _record_client_request("openai-chat", body)
    if entry is not None:
        request.extensions[_CLIENT_REQUEST_ENTRY] = entry


def _chat_response_snapshot(
    response: Response, body: bytes | None, *, content_encoding: str | None, eof: bool,
) -> ModelClientResponse:
    content_type = response.headers.get("content-type", "").partition(";")[0].strip().lower()
    format = "sse" if content_type == "text/event-stream" else (
        "json" if content_type == "application/json" or content_type.endswith("+json") else "unknown"
    )
    complete = eof and body is not None
    capture_error = "response_capture_failed" if body is None else None
    if format == "sse" and body is not None:
        replica = Response(
            200, headers={"content-type": "text/event-stream", "content-encoding": content_encoding or ""},
            stream=ByteStream(body), request=Request("GET", "https://model-response.invalid"),
        )
        try:
            complete = any(event.data == "[DONE]" for event in EventSource(replica))
        except (DecodingError, SSEError, ImportError):
            complete = False
            capture_error = "response_decode_failed"
    return ModelClientResponse(
        body=body, status_code=response.status_code, format=format,
        content_encoding=content_encoding, complete=complete, capture_error=capture_error,
    )


class _CapturedChatResponseStream(AsyncByteStream):
    def __init__(self, response: Response, entry: _ClientRequestEntry) -> None:
        self._response = response
        self._inner = response.stream
        self._entry = entry
        self._parts: list[bytes] | None = []
        self._eof = False

    def _capture(self, part: bytes | None = None, *, buffered: bool = False) -> None:
        if not self._entry.active:
            self._parts = None
            return
        try:
            if part is not None:
                if self._parts is not None:
                    self._parts.append(part)
                return
            body = self._response.content if buffered else (
                b"".join(self._parts) if self._parts is not None else None
            )
            self._entry.record_response(_chat_response_snapshot(
                self._response, body,
                content_encoding=None if buffered else self._response.headers.get("content-encoding") or None,
                eof=buffered or self._eof,
            ))
        except Exception:
            # Observation failures must not change SDK reads or cleanup errors.
            self._parts = None
            if part is None:
                try:
                    self._entry.record_response(ModelClientResponse(
                        body=None, status_code=self._response.status_code, format="unknown",
                        content_encoding=None, complete=False, capture_error="response_capture_failed",
                    ))
                except Exception:
                    # An unavailable recorder cannot provide a replacement view.
                    pass
        finally:
            if part is None:
                self._parts = None

    async def __aiter__(self) -> AsyncIterator[bytes]:
        iterator = self._inner.__aiter__()
        try:
            async for part in iterator:
                self._capture(part)
                yield part
            self._eof = True
        finally:
            if isinstance(iterator, AsyncGenerator):
                await iterator.aclose()

    async def aclose(self) -> None:
        try:
            await self._inner.aclose()
        finally:
            # HTTPX retains elapsed timing on its own stream wrapper.
            self._response.stream = self._inner
            self._capture()


async def _capture_chat_response(response: Response) -> None:
    entry = response.request.extensions.pop(_CLIENT_REQUEST_ENTRY, None)
    if entry is None or not entry.active:
        return
    capture = _CapturedChatResponseStream(response, entry)
    if response.is_stream_consumed:
        # A custom transport may return a response it already read and decoded.
        capture._capture(buffered=True)
    else:
        response.stream = capture


class _CapturedOpenAIProvider(OpenAIProvider):
    """Own the injected HTTP client across nested and repeated model contexts."""

    def __init__(self, *, base_url: str | None, api_key: str | None) -> None:
        self._capture_client_lock = asyncio.Lock()
        self._capture_client_entries = 0
        self._capture_http_client = self._new_client()
        super().__init__(base_url=base_url, api_key=api_key, http_client=self._capture_http_client)
        self.client.max_retries = 0

    @staticmethod
    def _new_client() -> DefaultAsyncHttpxClient:
        # Match the provider's existing HTTP pool and no-redirect policy while
        # retaining the SDK's timeout defaults.
        return DefaultAsyncHttpxClient(
            limits=Limits(), follow_redirects=False,
            event_hooks={"request": [_capture_chat_request], "response": [_capture_chat_response]},
        )

    async def __aenter__(self) -> "_CapturedOpenAIProvider":
        async with self._capture_client_lock:
            if self._capture_http_client.is_closed:
                client = self._new_client()
                super().__init__(openai_client=self.client.with_options(http_client=client))
                self._capture_http_client = client
            self._capture_client_entries += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        async with self._capture_client_lock:
            if self._capture_client_entries == 0:
                return
            self._capture_client_entries -= 1
            if self._capture_client_entries == 0:
                await self._capture_http_client.aclose()


@dataclass(frozen=True, slots=True)
class _OpenAIModelBinding:
    route_id: str
    model: str
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

    input_capabilities: Sequence[ModelInputCapability] | None = None
    unknown_inputs: UnknownModelInputs | None = None

    def __post_init__(self) -> None:
        model = self.model.strip().removeprefix("openai:")
        if not self.route_id.strip() or not model:
            raise ValueError("OpenAI model binding is incomplete")
        capabilities = normalize_input_capabilities(
            () if self.input_capabilities is None else self.input_capabilities,
            resolve_transport=lambda mime: openai_input_transport(mime, "binary", model),
        )
        unknown = self.unknown_inputs if self.unknown_inputs is not None else "reject"
        if unknown not in {"reject", "passthrough"}:
            raise ValueError("OpenAI unknown_inputs policy is invalid")
        object.__setattr__(self, "input_capabilities", capabilities)
        object.__setattr__(self, "unknown_inputs", unknown)
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
    def contract(self) -> "dict[str, JsonValue]":
        settings: dict[str, JsonValue] = {}
        if self.max_tokens is not None:
            settings["max_tokens"] = self.max_tokens
        result: dict[str, JsonValue] = {
            "provider": self.provider,
            "model_identity": self.model_identity,
            "settings": settings,
        }
        result["inputs"] = {
            "endpoint": "openai-chat",
            "unknown": self.unknown_inputs,
            "capabilities": [item.to_payload() for item in self.input_capabilities],
        }
        return result

    def materialize(self) -> Model:
        connection = _resolved_connection(self)
        try:
            provider = _CapturedOpenAIProvider(
                base_url=connection["base_url"],
                api_key=connection["api_key"],
            )

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
            input_binding=self,
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
        input_binding: ModelBinding | None = None,
    ) -> None:
        super().__init__(wrapped)
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._input_binding = input_binding

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        if self._input_binding is not None:
            validate_model_inputs(self._input_binding, messages)
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
        if self._input_binding is not None:
            validate_model_inputs(self._input_binding, messages)
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
