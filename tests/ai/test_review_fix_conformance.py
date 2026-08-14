#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression checks for the current runtime composition boundary."""

from linktools.ai.model import ModelConnectionConfig, ModelRoute, OpenAIModelMaterializer, StaticModelCredentialProvider
from linktools.ai import RuntimeStorage


def test_model_connection_configuration_is_secret_free_and_stable() -> None:
    connection = ModelConnectionConfig("connection", "https://example.test/v1", 10.0, "credential-a")
    assert connection.base_url == "https://example.test/v1"
    model = OpenAIModelMaterializer(StaticModelCredentialProvider({"credential-a": "secret"})).materialize(
        ModelRoute("route", "openai", "openai:gpt-test"), connection
    )
    assert "secret" not in repr(model)


def test_runtime_storage_normalizes_sqlite_paths() -> None:
    storage = RuntimeStorage.sqlite("relative.db")
    assert storage.target_path is not None and storage.target_path.as_posix().endswith("/relative.db")
