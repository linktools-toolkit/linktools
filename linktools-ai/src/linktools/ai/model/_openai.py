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
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import merge_model_settings

from ._binding import _ProviderModelBinding
from ._http import HttpClientOwner, capture_request, capture_response


# Match each supported protocol's SDK request ownership without copying its encoding.
_COMMON_REQUEST_SETTING_FIELDS = {
    "temperature": "temperature", "top_p": "top_p", "parallel_tool_calls": "parallel_tool_calls",
    "service_tier": "service_tier", "openai_service_tier": "service_tier",
    "openai_user": "user", "openai_top_logprobs": "top_logprobs",
    "openai_store": "store", "openai_moderation": "moderation",
    "openai_prompt_cache_key": "prompt_cache_key",
    "openai_prompt_cache_retention": "prompt_cache_retention",
    "openai_prompt_cache_options": "prompt_cache_options",
}
_CHAT_REQUEST_SETTING_FIELDS = {
    **_COMMON_REQUEST_SETTING_FIELDS,
    "stop_sequences": "stop", "seed": "seed", "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty", "logit_bias": "logit_bias",
    "openai_reasoning_effort": "reasoning_effort", "openai_logprobs": "logprobs",
    "openai_prediction": "prediction", "openai_text_verbosity": "verbosity",
    "openai_continuous_usage_stats": "stream_options",
}
_RESPONSES_REQUEST_SETTING_FIELDS = {
    **_COMMON_REQUEST_SETTING_FIELDS,
    "max_tokens": "max_output_tokens", "openai_reasoning_effort": "reasoning",
    "openai_reasoning_mode": "reasoning", "openai_reasoning_context": "reasoning",
    "openai_reasoning_summary": "reasoning", "openai_text_verbosity": "text",
    "openai_truncation": "truncation", "openai_context_management": "context_management",
    "openai_background": "background", "openai_include_code_execution_outputs": "include",
    "openai_include_web_search_sources": "include", "openai_include_file_search_results": "include",
    "openai_prompt_cache_diagnostics": "prompt_cache_options",
}


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
        _ProviderModelBinding.validate_request_owners(
            merge_model_settings(self.settings, model_settings),
            wire_fields={**_CHAT_REQUEST_SETTING_FIELDS,
                         "max_tokens": ("max_completion_tokens", "max_tokens")},
            request_fields=("model", "messages", "tools", "tool_choice", "stream"),
        )
        settings, parameters = super().prepare_request(model_settings, model_request_parameters)
        if parameters.output_mode == "native" or (
            parameters.output_mode == "prompted" and self.profile.get("supports_json_object_output", False)
        ):
            _ProviderModelBinding.validate_request_owners(
                settings, wire_fields={}, request_fields=("response_format",),
            )
        if any(isinstance(tool, WebSearchTool) for tool in parameters.native_tools):
            _ProviderModelBinding.validate_request_owners(
                settings, wire_fields={}, request_fields=("web_search_options",),
            )
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
        _ProviderModelBinding.validate_request_owners(
            merge_model_settings(self.settings, model_settings),
            wire_fields=_RESPONSES_REQUEST_SETTING_FIELDS,
            request_fields=("model", "input", "instructions", "tools", "tool_choice", "stream",
                            "previous_response_id", "conversation"),
        )
        settings, parameters = super().prepare_request(model_settings, model_request_parameters)
        if parameters.output_mode == "native" or (
            parameters.output_mode == "prompted" and self.profile.get("supports_json_object_output", False)
        ):
            _ProviderModelBinding.validate_request_owners(
                settings, wire_fields={}, request_fields=("text",),
            )
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
