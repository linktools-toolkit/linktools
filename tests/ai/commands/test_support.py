#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import unittest
from unittest import mock

from linktools.cli import CommandError
from linktools.ai.model.registry import RuntimeModelConfig
from linktools.commands.ai.support import resolve_model_config, validate_session_id


class TestResolveModelConfig(unittest.TestCase):
    def test_flags_take_precedence_over_env(self):
        with mock.patch.dict(
            os.environ,
            {
                "OPENAI_MODEL": "env-model",
                "OPENAI_BASE_URL": "https://env.example.com",
                "OPENAI_API_KEY": "env-key",
            },
            clear=False,
        ):
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
        with mock.patch.dict(
            os.environ,
            {
                "OPENAI_MODEL": "env-model",
                "OPENAI_BASE_URL": "https://env.example.com",
                "OPENAI_API_KEY": "env-key",
            },
            clear=False,
        ):
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
                resolve_model_config(
                    model="m", base_url="https://example.com", api_key=None
                )


class TestValidateSessionId(unittest.TestCase):
    def test_normal_session_id_passes_through_unchanged(self):
        self.assertEqual(validate_session_id("main"), "main")
        self.assertEqual(validate_session_id("my-session_1"), "my-session_1")

    def test_empty_raises_command_error(self):
        with self.assertRaises(CommandError):
            validate_session_id("")

    def test_dot_raises_command_error(self):
        with self.assertRaises(CommandError):
            validate_session_id(".")

    def test_dotdot_raises_command_error(self):
        with self.assertRaises(CommandError):
            validate_session_id("..")

    def test_dotdot_segment_raises_command_error(self):
        with self.assertRaises(CommandError):
            validate_session_id("../evil")

    def test_forward_slash_raises_command_error(self):
        with self.assertRaises(CommandError):
            validate_session_id("a/b")

    def test_backslash_raises_command_error(self):
        with self.assertRaises(CommandError):
            validate_session_id("a\\b")


class TestBuildRuntimeLink(unittest.TestCase):
    """§2.8 / §22 CLI: build_runtime() wires the minimal Runtime link
    (storage + model router + execution) without touching private Storage
    directories or Runtime private fields."""

    def test_build_runtime_constructs_runtime(self):
        import tempfile
        from argparse import Namespace

        from linktools.ai.runtime import Runtime
        from linktools.ai.storage.facade import FileStorage

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch(
                "linktools.commands.ai.support.build_storage",
                lambda: FileStorage(root=tmp),
            ):
                from linktools.commands.ai.support import build_runtime

                args = Namespace(
                    model="test", base_url="https://x", api_key="k", workdir=tmp
                )
                rt = build_runtime(args)
            self.assertIsInstance(rt, Runtime)
            # The link is wired through the public API -- no private fields.
            self.assertFalse(hasattr(rt, "capability_assembler"))
            self.assertFalse(hasattr(rt, "assemble"))


if __name__ == "__main__":
    unittest.main()
