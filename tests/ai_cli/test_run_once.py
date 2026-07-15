#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``run_once`` exit-code contract (spec §22/§27.1).

0 = completed; 4 = paused for approval (run_id/approval_id printed); 130 =
Ctrl+C after cancelling the run through the runtime (not just the process).
Drives the console layer with ``FakeRuntimeClient`` so no real Runtime/model is
needed."""

import asyncio
import contextlib
import io
import json
import unittest

from linktools.ai_cli.client import FakeRuntimeClient
from linktools.ai_cli.console.run_once import run_once


def _call(fake, **overrides) -> int:
    kwargs = dict(
        prompt="hi",
        agent=None,
        session="main",
        base_url=None,
        model=None,
        api_key=None,
        json_output=False,
        client=fake,
    )
    kwargs.update(overrides)
    return run_once(**kwargs)


class TestRunOnceExitCodes(unittest.TestCase):
    def test_success_returns_zero(self):
        fake = FakeRuntimeClient(stream_events=[{"type": "text", "text": "hello"}])
        self.assertEqual(_call(fake), 0)

    def test_paused_returns_exit_code_4(self):
        fake = FakeRuntimeClient(
            stream_events=[{"type": "paused", "run_id": "r1", "approval_id": "a1"}]
        )
        self.assertEqual(_call(fake), 4)

    def test_interrupt_returns_130_and_cancels_run(self):
        fake = FakeRuntimeClient(
            stream_events=[{"type": "text", "text": "partial"}],
            stream_error=asyncio.CancelledError(),
        )
        code = _call(fake)
        self.assertEqual(code, 130)
        # The run minted its own id; cancel must target exactly that id.
        self.assertEqual(fake.cancel_calls, [fake.last_run_id])

    def test_missing_prompt_raises_command_error(self):
        from linktools.cli import CommandError

        fake = FakeRuntimeClient()
        with self.assertRaises(CommandError):
            _call(fake, prompt=None)

    def test_paused_json_mode_emits_event_line(self):
        # --json must surface the pause as a structured event, not just exit 4.
        fake = FakeRuntimeClient(
            stream_events=[{"type": "paused", "run_id": "r1", "approval_id": "a1"}]
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = _call(fake, json_output=True)
        self.assertEqual(code, 4)
        self.assertEqual(
            json.loads(buf.getvalue()),
            {"type": "paused", "run_id": "r1", "approval_id": "a1"},
        )


if __name__ == "__main__":
    unittest.main()
