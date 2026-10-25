#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenAI Chat and Responses model assembly."""

from types import TracebackType

from httpx2 import AsyncClient, Limits
try:
    from openai import DefaultAsyncHttpxClient
except ModuleNotFoundError as error:
    if error.name != "openai":
        raise
    raise ImportError('OpenAI model adapters require pip install "linktools-ai[openai]"') from error
from openai.types.chat import ChatCompletion
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models import Model, ModelRequestParameters, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from ._binding import _ProviderModelBinding
from ._http import HttpClientOwner, capture_request, capture_response


class _OutputOnlyOpenAIChatModel(OpenAIChatModel):
    def _validate_completion(self, response: ChatCompletion) -> ChatCompletion:
        # The SDK's documented validation hook runs before its first-choice access.
        response = super()._validate_completion(response)
        if not response.choices:
            raise UnexpectedModelBehavior("Chat completion returned no choices")
        return response

    def prepare_request(
        self, model_settings: ModelSettings | None, model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        settings, parameters = super().prepare_request(model_settings, model_request_parameters)
        return _ProviderModelBinding.prepare_output_tool_settings(
            settings, parameters,
            unsupported_settings=(
                *getattr(self, "unsupported_settings", ()),
                *self.profile.get("openai_unsupported_model_settings", ()),
            ),
        ), parameters


class _OutputOnlyOpenAIResponsesModel(OpenAIResponsesModel):
    def prepare_request(
        self, model_settings: ModelSettings | None, model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        settings, parameters = super().prepare_request(model_settings, model_request_parameters)
        return _ProviderModelBinding.prepare_output_tool_settings(
            settings, parameters,
            unsupported_settings=(
                *getattr(self, "unsupported_settings", ()),
                *self.profile.get("openai_unsupported_model_settings", ()),
            ),
        ), parameters


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
    def _build_model(self, settings: ModelSettings) -> Model:
        provider = _CapturedOpenAIProvider(base_url=self.base_url, api_key=self.api_key)
        model_type = _OutputOnlyOpenAIChatModel if self.api == "openai" else _OutputOnlyOpenAIResponsesModel
        return model_type(self.model, provider=provider, settings=settings or None,
                          profile=self._sdk_profile())


class _OpenAIResponsesModelBinding(_OpenAIModelBinding):
    __slots__ = ()
    api = "openai-responses"
