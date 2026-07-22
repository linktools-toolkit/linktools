#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``continue_run`` status dispatch.

WAITING_APPROVAL + --approve -> approve + resume; + --reject -> reject + cancel;
+ --resume -> resume; (no flag) -> interactive. RUNNING -> report; terminal ->
show status; missing run -> CommandError. Drives the console layer with
``FakeRuntimeClient``."""

import types
import unittest
from unittest import mock

from linktools.ai.run.models import RunStatus
from linktools.ai_cli.client import FakeRuntimeClient
from linktools.ai_cli.console.continue_run import continue_run


def _record(status) -> types.SimpleNamespace:
    return types.SimpleNamespace(status=status)


class TestContinueWaitingApproval(unittest.TestCase):
    def _fake(self, *, run_id="run-1", resume_events=None) -> FakeRuntimeClient:
        return FakeRuntimeClient(
            run_record=_record(RunStatus.WAITING_APPROVAL),
            resume_events=resume_events or [{"type": "resumed", "run_id": "run-1"}],
            approvals=[types.SimpleNamespace(id="a1", run_id="run-1")],
        )

    def test_approve_approves_then_resumes(self):
        fake = self._fake(resume_events=[{"type": "resumed", "run_id": "run-1"}])
        code = continue_run("run-1", approve=True, client=fake)
        self.assertEqual(code, 0)
        self.assertEqual(fake.approve_calls, ["a1"])
        self.assertEqual(fake.resume_calls, ["run-1"])
        self.assertEqual(fake.cancel_calls, [])

    def test_reject_rejects_then_cancels(self):
        fake = self._fake()
        code = continue_run("run-1", reject=True, client=fake)
        self.assertEqual(code, 0)
        self.assertEqual(fake.reject_calls, [("a1", None)])
        self.assertEqual(fake.cancel_calls, ["run-1"])
        self.assertEqual(fake.resume_calls, [])

    def test_resume_resumes_without_approving(self):
        fake = self._fake()
        code = continue_run("run-1", resume=True, client=fake)
        self.assertEqual(code, 0)
        self.assertEqual(fake.resume_calls, ["run-1"])
        self.assertEqual(fake.approve_calls, [])

    def test_resume_paused_again_returns_4(self):
        fake = self._fake(
            resume_events=[
                {"type": "resumed", "run_id": "run-1"},
                {"type": "paused", "run_id": "run-1", "approval_id": "a2"},
            ]
        )
        code = continue_run("run-1", resume=True, client=fake)
        self.assertEqual(code, 4)

    def test_interactive_approve_choice(self):
        fake = self._fake()
        with mock.patch("builtins.input", return_value="approve"):
            code = continue_run("run-1", client=fake)
        self.assertEqual(code, 0)
        self.assertEqual(fake.approve_calls, ["a1"])

    def test_interactive_later_keeps_pending(self):
        fake = self._fake()
        with mock.patch("builtins.input", return_value=""):
            code = continue_run("run-1", client=fake)
        self.assertEqual(code, 0)
        self.assertEqual(fake.approve_calls, [])
        self.assertEqual(fake.cancel_calls, [])

    def test_reject_builds_client_without_model(self):
        # `continue --reject` never drives the model, so it must NOT require one
        # . build_runtime_client is called with with_model=False.
        import linktools.ai_cli.console.continue_run as mod

        fake = self._fake()
        built: dict = {}

        def fake_builder(*, with_model=True, **_):
            built["with_model"] = with_model
            return fake

        with mock.patch.object(mod, "build_runtime_client", fake_builder):
            code = continue_run("run-1", reject=True)
        self.assertEqual(code, 0)
        self.assertFalse(built["with_model"])

    def test_approve_builds_client_with_model(self):
        # Approve resumes, which drives the model -> with_model=True.
        import linktools.ai_cli.console.continue_run as mod

        fake = self._fake()
        built: dict = {}

        def fake_builder(*, with_model=True, **_):
            built["with_model"] = with_model
            return fake

        with mock.patch.object(mod, "build_runtime_client", fake_builder):
            continue_run("run-1", approve=True)
        self.assertTrue(built["with_model"])


class TestContinueOtherStates(unittest.TestCase):
    def test_running_reports_and_does_not_resume(self):
        fake = FakeRuntimeClient(run_record=_record(RunStatus.RUNNING))
        code = continue_run("run-1", client=fake)
        self.assertEqual(code, 0)
        self.assertEqual(fake.resume_calls, [])

    def test_terminal_shows_status_and_does_not_resume(self):
        fake = FakeRuntimeClient(run_record=_record(RunStatus.SUCCEEDED))
        code = continue_run("run-1", client=fake)
        self.assertEqual(code, 0)
        self.assertEqual(fake.resume_calls, [])

    def test_missing_run_raises_command_error(self):
        from linktools.cli import CommandError

        fake = FakeRuntimeClient(run_record=None)
        with self.assertRaises(CommandError):
            continue_run("nope", client=fake)


if __name__ == "__main__":
    unittest.main()
