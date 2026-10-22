#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenAI Chat and Responses model assembly."""

from collections.abc import Mapping
from types import TracebackType

from httpx2 import AsyncClient, Limits
try:
    from openai import DefaultAsyncHttpxClient
except ModuleNotFoundError as error:
    if error.name != "openai":
        raise
    raise ImportError('OpenAI model adapters require pip install "linktools-ai[openai]"') from error
from pydantic_ai.models import Model, ModelSettings
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from ._binding import _ProviderModelBinding
from ._http import HttpClientOwner, capture_request, capture_response


class _CapturedOpenAIProvider(OpenAIProvider):
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
        super().__init__(openai_client=self.client.with_options(http_client=client))

    async def __aenter__(self) -> "_CapturedOpenAIProvider":
        await self._http_owner.acquire(self._replace_client)
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None,
                        exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        await self._http_owner.release()


class _OpenAIModelBinding(_ProviderModelBinding):
    __slots__ = ()
    _provider_type = OpenAIProvider
    _profile_type = OpenAIModelProfile
    def _build_model(self, connection: Mapping[str, object], settings: ModelSettings) -> Model:
        provider = _CapturedOpenAIProvider(base_url=connection["base_url"], api_key=connection["api_key"])
        model_type = OpenAIChatModel if self.api == "openai-chat" else OpenAIResponsesModel
        return model_type(self.model, provider=provider, settings=settings or None,
                          **({"profile": self._sdk_profile()} if self._profile_overrides else {}))


class _OpenAIResponsesModelBinding(_OpenAIModelBinding):
    __slots__ = ()
    api = "openai-responses"
