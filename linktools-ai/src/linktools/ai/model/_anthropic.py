#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Anthropic Messages model assembly."""

from types import TracebackType

try:
    from anthropic import DefaultAsyncHttpxClient
except ModuleNotFoundError as error:
    if error.name != "anthropic":
        raise
    raise ImportError('Anthropic model adapters require pip install "linktools-ai[anthropic]"') from error
from httpx2 import AsyncClient, Limits
from pydantic_ai.models import Model, ModelRequestParameters, ModelSettings
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.settings import merge_model_settings

from ._binding import _ProviderModelBinding
from ._http import HttpClientOwner, capture_request, capture_response


# These are ownership names, not a serializer or an extension allowlist. Unified
# thinking/cache remain SDK fallbacks; only explicit native controls own these objects.
_REQUEST_SETTING_FIELDS = {
    "max_tokens": "max_tokens", "temperature": "temperature", "top_p": "top_p",
    "top_k": "top_k", "stop_sequences": "stop_sequences",
    "service_tier": "service_tier", "anthropic_service_tier": "service_tier",
    "anthropic_metadata": "metadata", "anthropic_thinking": "thinking",
    "anthropic_effort": "output_config", "anthropic_task_budget": "output_config",
    "anthropic_speed": "speed",
    "anthropic_cache": "cache_control", "anthropic_cache_diagnostics": "diagnostics",
}
_REQUEST_FIELDS = (
    "model", "messages", "system", "tools", "tool_choice", "mcp_servers", "stream",
    "container", "context_management",
)


class _OutputOnlyAnthropicModel(AnthropicModel):
    def prepare_request(
        self, model_settings: ModelSettings | None, model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        _ProviderModelBinding.validate_request_owners(
            merge_model_settings(self.settings, model_settings),
            wire_fields=_REQUEST_SETTING_FIELDS, request_fields=_REQUEST_FIELDS,
        )
        settings, parameters = super().prepare_request(model_settings, model_request_parameters)
        if parameters.output_mode == "native":
            _ProviderModelBinding.validate_request_owners(
                settings, wire_fields={}, request_fields=("output_config",),
            )
        return _ProviderModelBinding.prepare_output_tool_settings(
            settings, parameters, unsupported_settings=getattr(self, "unsupported_settings", ()),
        ), parameters


class _CapturedAnthropicProvider(AnthropicProvider):
    def __init__(self, *, base_url: str | None, api_key: str | None) -> None:
        self._http_owner = HttpClientOwner(self._new_client)
        super().__init__(base_url=base_url, api_key=api_key, http_client=self._http_owner.client)
        self.client.max_retries = 0

    @staticmethod
    def _new_client() -> DefaultAsyncHttpxClient:
        return DefaultAsyncHttpxClient(
            limits=Limits(), follow_redirects=False,
            event_hooks={"request": [capture_request], "response": [capture_response]},
        )

    def _replace_client(self, client: AsyncClient) -> None:
        super().__init__(anthropic_client=self.client.with_options(http_client=client))

    async def __aenter__(self) -> "_CapturedAnthropicProvider":
        await self._http_owner.acquire(self._replace_client)
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None,
                        exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        await self._http_owner.release()


class _AnthropicModelBinding(_ProviderModelBinding):
    __slots__ = ()
    api = "anthropic"
    _provider = "anthropic"

    def _build_model(self, settings: ModelSettings) -> Model:
        provider = _CapturedAnthropicProvider(base_url=self.base_url, api_key=self.api_key)
        return _OutputOnlyAnthropicModel(self.model, provider=provider, settings=settings or None,
                                        profile=self._sdk_profile())
