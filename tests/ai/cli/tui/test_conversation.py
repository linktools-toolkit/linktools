#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conversation folder tests.

The folder turns streamed Runtime dict-events into structured turns. These are
pure-data tests (no Textual), covering the event-accumulation rules that the
conversation view renders."""

import unittest

from linktools.ai.cli.tui.conversation import Conversation


class ConversationFolderTests(unittest.TestCase):
    def test_user_prompt_appends_user_turn(self) -> None:
        conv = Conversation()
        conv.add_user("hello")
        self.assertEqual(conv.turns[0].kind, "user")
        self.assertEqual(conv.turns[0].text, "hello")

    def test_consecutive_text_events_accumulate_into_one_assistant_turn(self) -> None:
        conv = Conversation()
        conv.apply({"type": "text", "text": "Hello"})
        conv.apply({"type": "text", "text": ", world"})
        self.assertEqual(len(conv.turns), 1)
        self.assertEqual(conv.turns[0].kind, "assistant")
        self.assertEqual(conv.turns[0].text, "Hello, world")
        self.assertTrue(conv.turns[0].streaming)

    def test_text_after_a_tool_event_starts_new_assistant_turn(self) -> None:
        conv = Conversation()
        conv.apply({"type": "tool", "id": "t1", "name": "read_file", "phase": "start"})
        conv.apply({"type": "text", "text": "result"})
        self.assertEqual(len(conv.turns), 2)
        self.assertEqual(conv.turns[0].kind, "tool")
        self.assertEqual(conv.turns[1].kind, "assistant")

    def test_tool_events_with_same_call_id_fold_into_one_turn(self) -> None:
        conv = Conversation()
        conv.apply({"type": "tool", "id": "t1", "name": "bash", "phase": "start"})
        conv.apply(
            {
                "type": "tool",
                "id": "t1",
                "name": "bash",
                "phase": "end",
                "ok": True,
                "detail": "0",
            }
        )
        self.assertEqual(len(conv.turns), 1)
        self.assertEqual(conv.turns[0].status, "ok")
        self.assertEqual(conv.turns[0].detail, "0")

    def test_failed_event_becomes_error_turn(self) -> None:
        conv = Conversation()
        conv.apply({"type": "failed", "error_type": "ValueError", "message": "bad"})
        self.assertEqual(conv.turns[0].kind, "error")
        self.assertIn("ValueError", conv.turns[0].message)

    def test_cancelled_and_resumed_become_status_turns(self) -> None:
        conv = Conversation()
        conv.apply({"type": "cancelled"})
        conv.apply({"type": "resumed"})
        self.assertEqual(conv.turns[0].kind, "status")
        self.assertEqual(conv.turns[1].kind, "status")

    def test_clear_resets_turns(self) -> None:
        conv = Conversation()
        conv.add_user("a")
        conv.clear()
        self.assertEqual(conv.turns, ())

    def test_unknown_event_type_ignored(self) -> None:
        conv = Conversation()
        conv.apply({"type": "future_event_kind"})
        self.assertEqual(conv.turns, ())

    def test_load_from_turns_rebuilds_user_and_assistant(self) -> None:
        conv = Conversation()
        per_turn = (
            (
                {"kind": "request", "parts": [{"type": "text", "content": "hello"}]},
                {
                    "kind": "response",
                    "parts": [{"type": "text", "content": "hi there"}],
                },
            ),
        )
        conv.load_from_turns(per_turn)
        kinds = [t.kind for t in conv.turns]
        assert "user" in kinds
        assert "assistant" in kinds
        assert any(t.kind == "assistant" and "hi there" in t.text for t in conv.turns)

    def test_load_from_turns_skips_non_text_parts(self) -> None:
        conv = Conversation()
        per_turn = (
            (
                {
                    "kind": "response",
                    "parts": [
                        {"type": "tool_call", "name": "x"},
                        {"type": "text", "content": "ok"},
                    ],
                },
            ),
        )
        conv.load_from_turns(per_turn)
        assert len(conv.turns) == 1
        assert conv.turns[0].kind == "assistant"

    def test_load_from_turns_clears_existing(self) -> None:
        conv = Conversation()
        conv.add_user("old")
        conv.load_from_turns(
            (({"kind": "request", "parts": [{"type": "text", "content": "new"}]},),)
        )
        assert len(conv.turns) == 1
        assert conv.turns[0].text == "new"


if __name__ == "__main__":
    unittest.main()
