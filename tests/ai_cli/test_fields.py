#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ConfigField resolution tests (spec §30).

Priority: explicit → env → project ``.linktools/config.yaml`` → JSON cache →
interactive prompt → error. Base URL/model are cached; the API key is never
cached (and is marked secret). Prompts are simulated by patching the prompt
function so no real stdin is needed."""

import contextlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from linktools.core._config_store import ConfigStore
from linktools.rich import set_no_input

from linktools.ai_cli import fields
from linktools.ai_cli.errors import MissingConfigError
from linktools.ai_cli.prompt_provider import resolve_ai_config

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


def _fake_prompt(message, **_kwargs):
    m = message.lower()
    if "base url" in m:
        return "https://prompted.example.com"
    if "model" in m:
        return "prompted-model"
    if "api key" in m:
        return "sk-prompted"
    return "x"


def _store(tmp) -> ConfigStore:
    return ConfigStore(Path(tmp) / "cache.json")


class TestResolvePriority(unittest.TestCase):
    def tearDown(self) -> None:
        set_no_input(False)  # resolution flips the global no-input flag

    def test_explicit_overrides_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            with ai_env(OPENAI_BASE_URL="https://env", OPENAI_API_KEY="k"):
                resolved = resolve_ai_config(
                    base_url="https://explicit",
                    api_key="k",
                    cache_store=_store(tmp),
                    interactive=False,
                )
        self.assertEqual(resolved.base_url, "https://explicit")

    def test_env_overrides_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.yaml"
            cfg.write_text("openai:\n  base_url: https://project\n  api_key: pk\n")
            with ai_env(OPENAI_BASE_URL="https://env", OPENAI_API_KEY="ek"):
                resolved = resolve_ai_config(
                    config_yaml_path=cfg, cache_store=_store(tmp), interactive=False
                )
        self.assertEqual(resolved.base_url, "https://env")

    def test_project_overrides_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.yaml"
            cfg.write_text("openai:\n  base_url: https://project\n  api_key: pk\n")
            store = _store(tmp)
            store.set("ai.OPENAI_BASE_URL", "https://cached")
            with ai_env():
                resolved = resolve_ai_config(
                    config_yaml_path=cfg, cache_store=store, interactive=False
                )
        self.assertEqual(resolved.base_url, "https://project")

    def test_cache_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            store.set("ai.OPENAI_BASE_URL", "https://cached")
            store.set("ai.OPENAI_API_KEY", "ck")
            with ai_env():
                resolved = resolve_ai_config(cache_store=store, interactive=False)
        self.assertEqual(resolved.base_url, "https://cached")
        self.assertEqual(resolved.api_key, "ck")

    def test_non_interactive_missing_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with ai_env():
                with self.assertRaises(MissingConfigError):
                    resolve_ai_config(cache_store=_store(tmp), interactive=False)

    def test_explicit_does_not_overwrite_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            store.set("ai.OPENAI_BASE_URL", "https://cached")
            with ai_env():
                resolved = resolve_ai_config(
                    base_url="https://explicit",
                    api_key="k",
                    cache_store=store,
                    interactive=False,
                )
            reloaded = ConfigStore(Path(tmp) / "cache.json")
        self.assertEqual(resolved.base_url, "https://explicit")
        self.assertEqual(
            reloaded.get("ai.OPENAI_BASE_URL"), "https://cached"
        )  # cache untouched


class TestPromptAndCaching(unittest.TestCase):
    def tearDown(self) -> None:
        set_no_input(False)

    def test_prompted_values_are_cached(self):
        # First (interactive) resolve prompts and writes base_url/model to cache;
        # a second resolve reuses the cache for base_url/model without prompting.
        # (The API key is never cached, so it is supplied explicitly below.)
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "cache.json"
            with ai_env():
                with mock.patch("linktools.rich.prompt", side_effect=_fake_prompt):
                    first = resolve_ai_config(
                        cache_store=ConfigStore(store_path), interactive=True
                    )
                # Cache hit for base_url/model; prompt must NOT fire.
                with mock.patch(
                    "linktools.rich.prompt", side_effect=AssertionError("cache miss")
                ):
                    second = resolve_ai_config(
                        api_key="k",
                        cache_store=ConfigStore(store_path),
                        interactive=True,
                    )
            self.assertEqual(first.base_url, "https://prompted.example.com")
            self.assertEqual(first.model, "prompted-model")
            self.assertEqual(second.base_url, first.base_url)

    def test_api_key_is_not_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "cache.json"
            with ai_env():
                with mock.patch("linktools.rich.prompt", side_effect=_fake_prompt):
                    resolve_ai_config(
                        cache_store=ConfigStore(store_path), interactive=True
                    )
            raw = store_path.read_text()
        self.assertIn("https://prompted.example.com", raw)  # base_url cached
        self.assertIn("prompted-model", raw)  # model cached
        self.assertNotIn("sk-prompted", raw)  # api_key NOT cached


class TestFieldShape(unittest.TestCase):
    def test_api_key_is_secret_others_are_not(self):
        self.assertTrue(fields.OPENAI_API_KEY.secret)
        self.assertFalse(fields.OPENAI_BASE_URL.secret)
        self.assertFalse(fields.OPENAI_MODEL.secret)

    def test_api_key_value_is_redacted(self):
        from linktools.core._config import redact_config_value

        self.assertEqual(redact_config_value(fields.OPENAI_API_KEY, "sk-secret"), "***")
        self.assertEqual(
            redact_config_value(fields.OPENAI_BASE_URL, "https://x"), "https://x"
        )


if __name__ == "__main__":
    unittest.main()
