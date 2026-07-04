#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import unittest
from unittest import mock

from linktools.cli import CommandError
from linktools.ai.core.model_runtime import RuntimeModelConfig
from linktools.commands.ai.support import resolve_model_config


class TestResolveModelConfig(unittest.TestCase):

    def test_flags_take_precedence_over_env(self):
        with mock.patch.dict(os.environ, {
            "OPENAI_MODEL": "env-model",
            "OPENAI_BASE_URL": "https://env.example.com",
            "OPENAI_API_KEY": "env-key",
        }, clear=False):
            config = resolve_model_config(
                model="flag-model",
                base_url="https://flag.example.com",
                api_key="flag-key",
            )
        self.assertIsInstance(config, RuntimeModelConfig)
        self.assertEqual(config.model, "flag-model")
        self.assertEqual(config.base_url, "https://flag.example.com")
        self.assertEqual(config.api_key, "flag-key")
        self.assertEqual(config.protocol, "openai")
        self.assertEqual(config.model_type, "standard")

    def test_falls_back_to_env_when_flags_absent(self):
        with mock.patch.dict(os.environ, {
            "OPENAI_MODEL": "env-model",
            "OPENAI_BASE_URL": "https://env.example.com",
            "OPENAI_API_KEY": "env-key",
        }, clear=False):
            config = resolve_model_config(model=None, base_url=None, api_key=None)
        self.assertEqual(config.model, "env-model")
        self.assertEqual(config.base_url, "https://env.example.com")
        self.assertEqual(config.api_key, "env-key")

    def test_missing_base_url_raises_command_error(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(CommandError):
                resolve_model_config(model="m", base_url=None, api_key="k")

    def test_missing_api_key_raises_command_error(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(CommandError):
                resolve_model_config(model="m", base_url="https://example.com", api_key=None)


if __name__ == '__main__':
    unittest.main()
