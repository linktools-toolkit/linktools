#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable model semantics, connection resolution and bounded transport retries."""

import asyncio
import math
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import InitVar, dataclass, field
from typing import ClassVar

from pydantic import TypeAdapter
from urllib.parse import urlsplit, urlunsplit

from linktools.core import environ
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UserError
from pydantic_ai.models import Model, ModelMessage, ModelRequestParameters, ModelResponse, ModelSettings, StreamedResponse, RunContext as PydanticRunContext
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.profiles import DEFAULT_PROFILE, merge_profile

from ..core import ImmutableJsonMapping, JsonValue, canonical_json_bytes, normalize_json_value
from ..errors import AIError, ErrorCode
from ._contract import ModelBinding
from ._input import expand_input_types, validate_model_inputs

_logger = environ.get_logger("ai.model.binding")

@dataclass(frozen=True, slots=True)
class _ProviderModelBinding:
    api: ClassVar[str] = "openai"
    _provider: ClassVar[str] = "openai"
    _profile_type: ClassVar[type]
    _provider_type: ClassVar[type]
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

    profile: InitVar[Mapping[str, JsonValue] | None] = None
    _profile_overrides: ImmutableJsonMapping = field(init=False)
    input_types: InitVar[Sequence[str]] = ()
    _input_mime_types: tuple[str, ...] = field(init=False)

    def __post_init__(self, profile: Mapping[str, JsonValue] | None, input_types: Sequence[str]) -> None:
        if profile is not None and not isinstance(profile, Mapping):
            raise TypeError("profile must be a JSON mapping")
        declared = normalize_json_value(dict(profile) if profile is not None else {})
        object.__setattr__(self, "_profile_overrides", ImmutableJsonMapping(declared))
        self._sdk_profile()
        model = self.model.strip().removeprefix(f"{self.provider}:")
        if not self.route_id.strip() or not model:
            raise ValueError("model binding is incomplete")
        object.__setattr__(self, "_input_mime_types", expand_input_types(input_types))
        if self.connection_resolver is not None and not callable(
            self.connection_resolver
        ):
            raise ValueError("model connection resolver must be callable")
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
        return self._provider

    @property
    def model_identity(self) -> str:
        return f"{self.provider}:{self.model}"

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
        result["input_mime_types"] = list(self._input_mime_types)
        if self.api != "openai":
            result["api"] = self.api
        if self._profile_overrides:
            result["profile"] = dict(self._profile_overrides)
        return result

    def thinking_control(self) -> Mapping[str, JsonValue]:
        profile = merge_profile(DEFAULT_PROFILE, self._provider_type.model_profile(self.model) or {},
                                self._sdk_profile() or {})
        mode = "always_on" if profile.get("thinking_always_enabled") else (
            "controllable" if profile.get("supports_thinking") else "undeclared"
        )
        return {"mode": mode, "enabled_by_default": bool(profile.get("thinking_enabled_by_default"))}

    def materialize(self) -> Model:
        connection = _resolved_connection(self)
        try:
            settings: ModelSettings = {}
            if connection["timeout"] is not None:
                settings["timeout"] = connection["timeout"]
            if self.max_tokens is not None:
                settings["max_tokens"] = self.max_tokens

            model = self._build_model(connection, settings)
        except UserError as error:
            raise AIError(
                ErrorCode.MODEL_CONFIG_INVALID,
                retryable=False,
                safe_details={
                    "provider": self.provider,
                    "reason": "provider_configuration_invalid",
                },
            ) from error
        _logger.debug(
            "Model materialized: route=%s model=%s credential=%s",
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


    def _sdk_profile(self) -> dict[str, object] | None:
        if not self._profile_overrides:
            return None
        unknown = set(self._profile_overrides) - set(self._profile_type.__annotations__)
        if unknown:
            raise ValueError("profile contains unknown SDK keys: " + ", ".join(sorted(unknown)))
        return TypeAdapter(self._profile_type).validate_json(
            canonical_json_bytes(self._profile_overrides), strict=True,
        )

    def _build_model(self, connection: Mapping[str, object], settings: ModelSettings) -> Model:
        raise NotImplementedError

def _resolved_connection(binding: _ProviderModelBinding) -> dict[str, object]:
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
