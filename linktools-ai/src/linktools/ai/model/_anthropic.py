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
from pydantic_ai.models import Model, ModelSettings
from pydantic_ai.profiles.anthropic import AnthropicModelProfile
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from ._binding import _ProviderModelBinding
from ._http import HttpClientOwner, capture_request, capture_response


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
    _provider_type = AnthropicProvider
    _profile_type = AnthropicModelProfile
    api = "anthropic"
    _provider = "anthropic"

    def _build_model(self, settings: ModelSettings) -> Model:
        provider = _CapturedAnthropicProvider(base_url=self.base_url, api_key=self.api_key)
        return AnthropicModel(self.model, provider=provider, settings=settings or None,
                              profile=self._sdk_profile())
