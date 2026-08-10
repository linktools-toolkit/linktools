#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model deployment materialization boundary."""

from typing import Protocol

from linktools.core import environ
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from ..errors import AIError, ErrorCode
from ._connection import ModelConnectionConfig, ModelCredentialProvider
from ._registry import ModelRoute

_logger = environ.get_logger("ai.model.materializer")


class ModelMaterializer(Protocol):
    def materialize(self, route: ModelRoute, connection: "ModelConnectionConfig | None") -> Model: ...


class OpenAIModelMaterializer(ModelMaterializer):
    def __init__(self, credentials: ModelCredentialProvider) -> None:
        if credentials is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        self._credentials = credentials

    def materialize(self, route: ModelRoute, connection: "ModelConnectionConfig | None") -> Model:
        if route.provider != "openai":
            raise AIError(ErrorCode.MODEL_CONNECTION_UNSUPPORTED)
        api_key: "str | None" = None
        base_url: "str | None" = None
        settings: "ModelSettings | None" = None
        if connection is not None:
            base_url = connection.base_url
            if connection.credential_id is not None:
                api_key = self._credentials.get_api_key(connection.credential_id)
                if not isinstance(api_key, str) or not api_key.strip():
                    raise AIError(
                        ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
                        safe_details={"credential_id": connection.credential_id},
                    )
            if connection.timeout_seconds is not None:
                settings = {"timeout": connection.timeout_seconds}
        model = OpenAIChatModel(
            route.model.removeprefix("openai:"),
            provider=OpenAIProvider(base_url=base_url, api_key=api_key),
            settings=settings,
        )
        _logger.debug(
            "OpenAI model materialized: route=%s connection=%s credential=%s timeout=%s",
            route.route_id,
            connection.connection_id if connection is not None else None,
            connection.credential_id is not None if connection is not None else False,
            connection.timeout_seconds if connection is not None else None,
        )
        return model


__all__ = ["ModelMaterializer", "OpenAIModelMaterializer"]
