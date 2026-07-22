#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCP ``${ENV}`` expansion tests."""

import os
import unittest
from unittest import mock

from linktools.ai.errors import InvalidSpecError
from linktools.ai.mcp.env import expand_env_mapping, expand_env_value


class TestMcpEnvExpansion(unittest.TestCase):
    def test_expands_set_variable(self):
        with mock.patch.dict(os.environ, {"GITHUB_TOKEN": "tok-123"}, clear=False):
            self.assertEqual(expand_env_value("${GITHUB_TOKEN}"), "tok-123")

    def test_expands_within_larger_string(self):
        with mock.patch.dict(os.environ, {"NAME": "world"}, clear=False):
            self.assertEqual(expand_env_value("hello ${NAME}!"), "hello world!")

    def test_missing_env_fails(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(InvalidSpecError):
                expand_env_value("${MISSING_VAR}")

    def test_empty_env_fails(self):
        # An explicitly-empty variable is treated as unset (spec: fail closed).
        with mock.patch.dict(os.environ, {"EMPTY": ""}, clear=True):
            with self.assertRaises(InvalidSpecError):
                expand_env_value("${EMPTY}")

    def test_malformed_reference_rejected(self):
        # Lowercase / non-POSIX names are rejected, not silently passed through.
        for bad in ["${lower}", "${}", "${1ABC}"]:
            with self.assertRaises(InvalidSpecError):
                expand_env_value(bad)

    def test_non_env_string_passes_through(self):
        self.assertEqual(expand_env_value("plain literal"), "plain literal")
        self.assertEqual(expand_env_value(42), 42)

    def test_mapping_expansion(self):
        env = {
            "GITHUB_TOKEN": "${GITHUB_TOKEN}",
            "STATIC": "keep-me",
            "NESTED": ["${GITHUB_TOKEN}", "x"],
        }
        with mock.patch.dict(os.environ, {"GITHUB_TOKEN": "tok"}, clear=False):
            expanded = expand_env_mapping(env)
        self.assertEqual(expanded["GITHUB_TOKEN"], "tok")
        self.assertEqual(expanded["STATIC"], "keep-me")
        self.assertEqual(expanded["NESTED"], ["tok", "x"])
        # Input not mutated.
        self.assertEqual(env["GITHUB_TOKEN"], "${GITHUB_TOKEN}")

    def test_empty_or_none_env(self):
        self.assertEqual(expand_env_mapping(None), {})
        self.assertEqual(expand_env_mapping({}), {})


if __name__ == "__main__":
    unittest.main()
