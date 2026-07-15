#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stable configuration fields for ``lt ai`` (the model/endpoint config that is
stable across runs and worth caching).

Each field is a real ``linktools.core.ConfigField``. Required fields chain a
cached ``PromptProvider`` (ask once, remember) with an ``ErrorProvider`` so a
non-interactive invocation fails fast instead of blocking on a prompt. The API
key is ``secret=True`` and deliberately ``cached=False`` -- it is read from the
environment or a one-off secure prompt, never written to the JSON cache."""

from linktools.core import ConfigField, ErrorProvider, PromptProvider

# Field names mirror their environment variables so an empty-prefix
# EnvironmentSource reads them directly; the JSON cache keys by the same names.
OPENAI_BASE_URL = ConfigField.chain(
    PromptProvider("OpenAI Base URL", cached=True),
    ErrorProvider(
        "no OpenAI base URL: pass --base-url, set OPENAI_BASE_URL, "
        "or add `openai.base_url` to .linktools/config.yaml"
    ),
    name="OPENAI_BASE_URL",
    cast=str,
    required=True,
)

OPENAI_MODEL = ConfigField(
    name="OPENAI_MODEL",
    cast=str,
    provider=PromptProvider("Default model", default="", cached=True),
)

OPENAI_API_KEY = ConfigField.chain(
    PromptProvider("OpenAI API Key", password=True, cached=False),
    ErrorProvider("no OpenAI API key: pass --api-key or set OPENAI_API_KEY"),
    name="OPENAI_API_KEY",
    cast=str,
    required=True,
    secret=True,
)

# Optional for now (used once HttpRuntimeClient lands); resolved from env /
# project config / cache when present, otherwise None -- never prompted.
REMOTE_RUNTIME_URL = ConfigField(
    name="LINKTOOLS_AI_RUNTIME_URL",
    cast=str,
    default=None,
)

ALL_FIELDS = (OPENAI_BASE_URL, OPENAI_MODEL, OPENAI_API_KEY, REMOTE_RUNTIME_URL)
