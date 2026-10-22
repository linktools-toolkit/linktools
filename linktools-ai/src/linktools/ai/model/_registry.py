#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thread-safe model registry and immutable resolver captures."""

from collections.abc import Callable, Mapping, Sequence
from threading import RLock
from types import MappingProxyType

from linktools.core import environ
from pydantic_ai.models import Model

from ..core import JsonValue
from ..errors import AIError, ErrorCode
from ._contract import ModelBinding, ModelResolver, model_thinking_control

_logger = environ.get_logger("ai.model.registry")


class ModelRegistry:
    def __init__(self) -> None:
        self._bindings: dict[str, ModelBinding] = {}
        self._lock = RLock()

    @classmethod
    def create(
        cls,
        *,
        protocol: str,
        model: str,
        input_types: Sequence[str] = (),
        profile: Mapping[str, JsonValue] | None = None,
        base_url: "str | None" = None,
        api_key: "str | None" = None,
        timeout: "int | float | None" = None,
        max_retries: int = 2,
        retry_delay: "int | float" = 1.0,
        max_tokens: "int | None" = None,
        connection_resolver: "Callable[[], Mapping[str, object]] | None" = None,
    ) -> "ModelRegistry":
        """Create a default route using a supported protocol's dedicated factory.

        This constructs a new registry; register(binding) adds a binding to an
        existing registry. Protocol names are exact, not provider/model names.
        """
        factories = {
            "openai": cls.openai,
            "openai-responses": cls.openai_responses,
            "anthropic": cls.anthropic,
        }
        if not isinstance(protocol, str) or protocol not in factories:
            raise AIError(
                ErrorCode.MODEL_CONFIG_INVALID,
                "Unsupported model protocol; expected openai, openai-responses or anthropic",
                retryable=False,
                safe_details={"reason": "unsupported_model_protocol"},
            )
        return factories[protocol](
            model=model,
            input_types=input_types,
            profile=profile,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            max_tokens=max_tokens,
            connection_resolver=connection_resolver,
        )

    @classmethod
    def openai(
        cls,
        *,
        model: str,
        input_types: Sequence[str] = (),
        profile: Mapping[str, JsonValue] | None = None,
        base_url: "str | None" = None,
        api_key: "str | None" = None,
        timeout: "int | float | None" = None,
        max_retries: int = 2,
        retry_delay: "int | float" = 1.0,
        max_tokens: "int | None" = None,
        connection_resolver: "Callable[[], Mapping[str, object]] | None" = None,
    ) -> "ModelRegistry":
        """Create the default Chat route.

        input_types accepts image/audio/document/video SDK categories and
        mime:<MIME type> patterns. Only their expanded MIME types are allowed;
        provider encoding checks and resource budgets still apply.
        """
        registry = cls()
        registry.register_openai(
            "default",
            model=model,
            input_types=input_types,
            profile=profile,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            max_tokens=max_tokens,
            connection_resolver=connection_resolver,
        )
        return registry

    def register(self, binding: ModelBinding) -> None:
        if not binding.route_id.strip():
            raise ValueError("model route_id is required")
        with self._lock:
            self._bindings[binding.route_id] = binding
            _logger.info("model binding registered: route=%s", binding.route_id)

    def register_openai(
        self,
        route_id: str,
        *,
        model: str,
        input_types: Sequence[str] = (),
        profile: Mapping[str, JsonValue] | None = None,
        base_url: "str | None" = None,
        api_key: "str | None" = None,
        timeout: "int | float | None" = None,
        max_retries: int = 2,
        retry_delay: "int | float" = 1.0,
        max_tokens: "int | None" = None,
        connection_resolver: "Callable[[], Mapping[str, object]] | None" = None,
    ) -> None:
        from ._openai import _OpenAIModelBinding

        self.register(
            _OpenAIModelBinding(
                route_id=route_id,
                model=model,
                input_types=input_types,
                profile=profile,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
                max_tokens=max_tokens,
                connection_resolver=connection_resolver,
            )
        )

    @classmethod
    def openai_responses(
        cls,
        *,
        model: str,
        input_types: Sequence[str] = (),
        profile: Mapping[str, JsonValue] | None = None,
        base_url: "str | None" = None,
        api_key: "str | None" = None,
        timeout: "int | float | None" = None,
        max_retries: int = 2,
        retry_delay: "int | float" = 1.0,
        max_tokens: "int | None" = None,
        connection_resolver: "Callable[[], Mapping[str, object]] | None" = None,
    ) -> "ModelRegistry":
        """Create the default openai_responses route.

        input_types accepts image/audio/document/video SDK categories and
        mime:<MIME type> patterns. Only their expanded MIME types are allowed;
        provider encoding checks and resource budgets still apply.
        """
        registry = cls()
        registry.register_openai_responses(
            "default",
            model=model,
            input_types=input_types,
            profile=profile,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            max_tokens=max_tokens,
            connection_resolver=connection_resolver,
        )
        return registry

    def register_openai_responses(
        self,
        route_id: str,
        *,
        model: str,
        input_types: Sequence[str] = (),
        profile: Mapping[str, JsonValue] | None = None,
        base_url: "str | None" = None,
        api_key: "str | None" = None,
        timeout: "int | float | None" = None,
        max_retries: int = 2,
        retry_delay: "int | float" = 1.0,
        max_tokens: "int | None" = None,
        connection_resolver: "Callable[[], Mapping[str, object]] | None" = None,
    ) -> None:
        from ._openai import _OpenAIResponsesModelBinding

        self.register(
            _OpenAIResponsesModelBinding(
                route_id=route_id,
                model=model,
                input_types=input_types,
                profile=profile,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
                max_tokens=max_tokens,
                connection_resolver=connection_resolver,
            )
        )

    @classmethod
    def anthropic(
        cls,
        *,
        model: str,
        input_types: Sequence[str] = (),
        profile: Mapping[str, JsonValue] | None = None,
        base_url: "str | None" = None,
        api_key: "str | None" = None,
        timeout: "int | float | None" = None,
        max_retries: int = 2,
        retry_delay: "int | float" = 1.0,
        max_tokens: "int | None" = None,
        connection_resolver: "Callable[[], Mapping[str, object]] | None" = None,
    ) -> "ModelRegistry":
        """Create the default anthropic route.

        input_types accepts image/audio/document/video SDK categories and
        mime:<MIME type> patterns. Only their expanded MIME types are allowed;
        provider encoding checks and resource budgets still apply.
        """
        registry = cls()
        registry.register_anthropic(
            "default",
            model=model,
            input_types=input_types,
            profile=profile,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            max_tokens=max_tokens,
            connection_resolver=connection_resolver,
        )
        return registry

    def register_anthropic(
        self,
        route_id: str,
        *,
        model: str,
        input_types: Sequence[str] = (),
        profile: Mapping[str, JsonValue] | None = None,
        base_url: "str | None" = None,
        api_key: "str | None" = None,
        timeout: "int | float | None" = None,
        max_retries: int = 2,
        retry_delay: "int | float" = 1.0,
        max_tokens: "int | None" = None,
        connection_resolver: "Callable[[], Mapping[str, object]] | None" = None,
    ) -> None:
        from ._anthropic import _AnthropicModelBinding

        self.register(
            _AnthropicModelBinding(
                route_id=route_id,
                model=model,
                input_types=input_types,
                profile=profile,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
                max_tokens=max_tokens,
                connection_resolver=connection_resolver,
            )
        )

    def register_alias(self, alias_id: str, target_route_id: str) -> None:
        """Register an immutable route alias to an existing binding."""
        if not isinstance(alias_id, str) or not alias_id.strip():
            raise ValueError("model alias is required")
        if not isinstance(target_route_id, str) or not target_route_id.strip():
            raise ValueError("model alias target is required")
        with self._lock:
            target = self._bindings.get(target_route_id)
            if target is None:
                raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND)
            while isinstance(target, _ModelAliasBinding):
                target = target._target
            self._bindings[alias_id] = _ModelAliasBinding(alias_id, target)
            _logger.info(
                "model binding alias registered: alias=%s target=%s",
                alias_id,
                target.route_id,
            )

    def remove(self, route_id: str) -> None:
        with self._lock:
            if route_id in self._bindings:
                del self._bindings[route_id]
                _logger.info("model binding removed: route=%s", route_id)

    def capture(self) -> ModelResolver:
        with self._lock:
            return _CapturedModelResolver(MappingProxyType(dict(self._bindings)))


class _CapturedModelResolver:
    def __init__(self, bindings: "Mapping[str, ModelBinding]") -> None:
        self._bindings = bindings

    def resolve(self, route_id: str) -> ModelBinding:
        try:
            return self._bindings[route_id]
        except KeyError as error:
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND) from error

    def restore(
        self,
        payload: "Mapping[str, JsonValue]",
        *,
        route_id: "str | None" = None,
    ) -> ModelBinding:
        if route_id is None:
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND)
        binding = self._bindings.get(route_id)
        if binding is None:
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND)
        if dict(binding.contract) != dict(payload):
            raise AIError(ErrorCode.AGENT_BINDING_UNAVAILABLE)
        return binding


__all__ = ["ModelRegistry"]


class _ModelAliasBinding:
    __slots__ = ("_route_id", "_target")

    def __init__(self, route_id: str, target: ModelBinding) -> None:
        self._route_id = route_id
        self._target = target

    def __setattr__(self, name: str, value: object) -> None:
        if name in self.__slots__ and hasattr(self, name):
            raise AttributeError("model alias binding is immutable")
        object.__setattr__(self, name, value)

    @property
    def route_id(self) -> str:
        return self._route_id

    @property
    def provider(self) -> str:
        return self._target.provider

    @property
    def model_identity(self) -> str:
        return self._target.model_identity

    @property
    def contract(self) -> Mapping[str, JsonValue]:
        return self._target.contract

    def thinking_control(self) -> Mapping[str, JsonValue]:
        return model_thinking_control(self._target)

    def materialize(self) -> Model:
        return self._target.materialize()
