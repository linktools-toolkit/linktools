#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Console renderer tests.

``announce_paused`` expands a pause event with the approval request's
tool/arguments/reason; the request is fetched by the caller through
``RuntimeClient.get_approval`` (so this renderer never touches Storage -- spec
). ``print_event`` renders streamed text/tool events and the ``--json`` line
form."""

import contextlib
import io
import json
import unittest

from linktools.ai_cli.console.renderer import announce_paused, print_event


class _FakeApprovalRequest:
    def __init__(self, tool_name, arguments, reason):
        self.tool_name = tool_name
        self.arguments = arguments
        self.reason = reason


class _RecordingLogger:
    def __init__(self):
        self.lines: "list[str]" = []

    def warning(self, message, *args):
        self.lines.append(message % args if args else message)

    def info(self, message, *args):
        self.lines.append(message % args if args else message)

    def error(self, message, *args):
        self.lines.append(message % args if args else message)


class TestAnnouncePaused(unittest.TestCase):
    def test_renders_tool_arguments_reason_and_ids(self):
        request = _FakeApprovalRequest(
            tool_name="terminal",
            arguments={"command": ["echo", "hi"]},
            reason="high-risk shell",
        )
        logger = _RecordingLogger()
        announce_paused(
            request, {"type": "paused", "run_id": "r1", "approval_id": "a1"}, logger
        )
        rendered = "\n".join(logger.lines)
        self.assertIn("tool: terminal", rendered)
        self.assertIn("'command': ['echo', 'hi']", rendered)
        self.assertIn("reason: high-risk shell", rendered)
        self.assertIn("run_id: r1", rendered)
        self.assertIn("approval_id: a1", rendered)
        self.assertIn("lt ai continue r1 --approve", rendered)

    def test_no_request_degrades_to_ids_only(self):
        logger = _RecordingLogger()
        announce_paused(
            None, {"type": "paused", "run_id": "r9", "approval_id": "a9"}, logger
        )
        rendered = "\n".join(logger.lines)
        self.assertIn("tool: ?", rendered)
        self.assertIn("run_id: r9", rendered)
        self.assertIn("approval_id: a9", rendered)


class TestPrintEvent(unittest.TestCase):
    def test_text_event_prints_streamed(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_event(
                {"type": "text", "text": "chunk"},
                json_output=False,
                logger=_RecordingLogger(),
            )
        self.assertEqual(buf.getvalue(), "chunk")

    def test_tool_event_logs_collapsed(self):
        logger = _RecordingLogger()
        print_event(
            {"type": "tool", "name": "read_file", "phase": "end", "ok": True},
            json_output=False,
            logger=logger,
        )
        self.assertIn("[tool: read_file end ok]", logger.lines)

    def test_json_mode_emits_one_json_line(self):
        buf = io.StringIO()
        event = {"type": "text", "text": "hi"}
        with contextlib.redirect_stdout(buf):
            print_event(event, json_output=True, logger=_RecordingLogger())
        self.assertEqual(json.loads(buf.getvalue()), event)


if __name__ == "__main__":
    unittest.main()
