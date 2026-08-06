#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Configuration fields shared by local AI command entry points."""

from linktools.core import ConfigField, ErrorProvider, PromptProvider


OPENAI_BASE_URL = ConfigField.chain(
    PromptProvider("OpenAI Base URL", cached=True),
    ErrorProvider(
        "no OpenAI base URL: pass --base-url, set OPENAI_BASE_URL, "
        "or run interactively to prompt"
    ),
    name="OPENAI_BASE_URL",
    aliases=("base_url",),
    cast=str,
    required=True,
)

OPENAI_MODEL = ConfigField(
    name="OPENAI_MODEL",
    aliases=("model",),
    cast=str,
    provider=PromptProvider("Default model", cached=True),
)

OPENAI_API_KEY = ConfigField.chain(
    PromptProvider("OpenAI API Key", password=True, cached=False),
    ErrorProvider("no OpenAI API key: pass --api-key or set OPENAI_API_KEY"),
    name="OPENAI_API_KEY",
    aliases=("api_key",),
    cast=str,
    required=True,
    secret=True,
)


__all__ = ["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"]
