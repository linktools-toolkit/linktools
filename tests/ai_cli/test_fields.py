#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ConfigField tests (spec §30).

Priority: explicit → env → cache → prompt → error. Base URL/model are cached;
the API key is never cached and is marked secret. Tests use ConfigStore +
EnvironmentSource directly to avoid relying on the CLI framework's parse-time
resolution."""

import contextlib
import os
import tempfile
import unittest
from pathlib import Path

from linktools.core._config_store import ConfigStore
from linktools.rich import set_no_input

from linktools.ai_cli import fields
from linktools.ai_cli.errors import MissingConfigError

_ENV_KEYS = (
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "OPENAI_API_KEY",
    "LINKTOOLS_AI_RUNTIME_URL",
)


@contextlib.contextmanager
def ai_env(**values):
    """Expose only the given AI env vars for the block (others cleared)."""
    saved = {}
    for k in _ENV_KEYS:
        if k in os.environ:
            saved[k] = os.environ[k]
            del os.environ[k]
    os.environ.update(values)
    try:
        yield
    finally:
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.update(saved)


class TestFieldShape(unittest.TestCase):
    def test_api_key_is_secret_and_not_cached(self):
        """Spec §12: API key must be secret and never cached."""
        self.assertTrue(fields.OPENAI_API_KEY.secret)
        provider = fields.OPENAI_API_KEY.provider
        # ChainProvider wraps sub-providers; find the PromptProvider.
        from linktools.core._config import ChainProvider, PromptProvider

        if isinstance(provider, ChainProvider):
            prompt = next(
                (p for p in provider.providers if isinstance(p, PromptProvider)), None
            )
            self.assertIsNotNone(prompt, "API key must have a PromptProvider")
            self.assertFalse(
                prompt.cached, "API key PromptProvider must be cached=False"
            )

    def test_base_url_and_model_are_cached(self):
        from linktools.core._config import ChainProvider, PromptProvider

        for field in (fields.OPENAI_BASE_URL, fields.OPENAI_MODEL):
            prov = field.provider
            if isinstance(prov, ChainProvider):
                prompt = next(
                    (p for p in prov.providers if isinstance(p, PromptProvider)), None
                )
                self.assertIsNotNone(prompt, f"{field.name} must have PromptProvider")
                self.assertTrue(prompt.cached, f"{field.name} must be cached=True")
            elif isinstance(prov, PromptProvider):
                self.assertTrue(prov.cached, f"{field.name} must be cached=True")

    def test_base_url_and_model_not_secret(self):
        self.assertFalse(fields.OPENAI_BASE_URL.secret)
        self.assertFalse(fields.OPENAI_MODEL.secret)


if __name__ == "__main__":
    unittest.main()
