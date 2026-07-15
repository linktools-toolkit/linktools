#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``validate_session_id`` guards against path-escape in session ids."""

import unittest

from linktools.ai_cli.client import validate_session_id
from linktools.cli import CommandError


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


if __name__ == "__main__":
    unittest.main()
