#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RuntimeClient tests.

``FakeRuntimeClient`` is the shared double both console and TUI tests drive, so
it must implement every protocol method. ``LocalRuntimeClient``
maps each protocol method onto the Runtime + Storage + registries without
re-implementing capability assembly."""

import asyncio
import types
import unittest
from unittest import mock

from linktools.ai.agent.approval import ApprovalStatus
from linktools.ai_cli.client import (
    DoctorReport,
    FakeRuntimeClient,
    LocalRuntimeClient,
    RunRequest,
)


def _run(coro):
    return asyncio.run(coro)


class TestFakeRuntimeClientCoversProtocol(unittest.TestCase):
    """-- the Fake implements + records every protocol method."""

    def _fake(self) -> FakeRuntimeClient:
        return FakeRuntimeClient(
            stream_events=[{"type": "text", "text": "hi"}],
            resume_events=[{"type": "resumed", "run_id": "r1"}],
            sessions=[types.SimpleNamespace(id="s1")],
            runs=[types.SimpleNamespace(id="r1")],
            approvals=[types.SimpleNamespace(id="a1", run_id="r1")],
            agents=("default", "reviewer"),
            skills=("code-review",),
            mcp_servers=("github",),
            inspection=types.SimpleNamespace(tool_descriptors=()),
            doctor_report=DoctorReport(),
            run_record=types.SimpleNamespace(status="running"),
            session_record=types.SimpleNamespace(id="s1"),
            approval=types.SimpleNamespace(
                id="a1", tool_name="terminal", arguments={}, reason="r"
            ),
        )

    def test_run_stream_replays_events_and_records_request(self):
        fake = self._fake()
        events = _run(
            self._collect(fake.run_stream(RunRequest(prompt="p", run_id="r9")))
        )
        self.assertEqual(events, [{"type": "text", "text": "hi"}])
        self.assertEqual(fake.run_requests, [RunRequest(prompt="p", run_id="r9")])
        self.assertEqual(fake.last_run_id, "r9")

    def test_resume_stream_records_and_replays(self):
        fake = self._fake()
        events = _run(self._collect(fake.resume_stream("r1")))
        self.assertEqual(fake.resume_calls, ["r1"])
        self.assertEqual(events, [{"type": "resumed", "run_id": "r1"}])

    def test_cancel_approve_reject_recorded(self):
        fake = self._fake()
        _run(fake.cancel("r1"))
        _run(fake.approve("a1"))
        _run(fake.reject("a2", reason="nope"))
        self.assertEqual(fake.cancel_calls, ["r1"])
        self.assertEqual(fake.approve_calls, ["a1"])
        self.assertEqual(fake.reject_calls, [("a2", "nope")])

    def test_listers_and_getters_return_canned_data(self):
        fake = self._fake()
        self.assertEqual(_run(fake.list_sessions()), [types.SimpleNamespace(id="s1")])
        self.assertEqual(_run(fake.list_runs()), [types.SimpleNamespace(id="r1")])
        self.assertEqual(
            _run(fake.list_approvals()),
            [types.SimpleNamespace(id="a1", run_id="r1")],
        )
        self.assertEqual(_run(fake.list_agents()), ("default", "reviewer"))
        self.assertEqual(_run(fake.list_skills()), ("code-review",))
        self.assertEqual(_run(fake.list_mcp_servers()), ("github",))
        self.assertIs(_run(fake.get_run("r1")), fake._run_record)
        self.assertIs(_run(fake.get_session("s1")), fake._session_record)
        self.assertIs(_run(fake.get_approval("a1")), fake._approval)
        self.assertIs(_run(fake.inspect("default")), fake._inspection)
        self.assertIs(_run(fake.doctor()), fake._doctor_report)

    @staticmethod
    async def _collect(agen):
        out = []
        async for event in agen:
            out.append(event)
        return out


class _FakeApprovalStore:
    def __init__(self, request):
        self._request = request
        self.approve = mock.AsyncMock()
        self.reject = mock.AsyncMock()

    async def get(self, _approval_id):
        return self._request


def _local_bundle(*, approval_request=None):
    """A CliRuntimeBundle-shaped double for LocalRuntimeClient mapping tests."""
    return types.SimpleNamespace(
        project=types.SimpleNamespace(
            default_agent="default", state_root="/nonexistent"
        ),
        runtime=types.SimpleNamespace(
            run_stream=mock.AsyncMock(),
            resume=mock.AsyncMock(),
            cancel=mock.AsyncMock(),
            inspect=mock.AsyncMock(return_value="INSPECTION"),
        ),
        storage=types.SimpleNamespace(
            approvals=_FakeApprovalStore(approval_request),
            sessions=types.SimpleNamespace(get=mock.AsyncMock(return_value=None)),
            runs=types.SimpleNamespace(get=mock.AsyncMock(return_value="RUN")),
        ),
        agents=types.SimpleNamespace(
            get=mock.AsyncMock(return_value="SPEC"),
            list_ids=mock.AsyncMock(return_value=("default", "reviewer")),
        ),
        skill_index=types.SimpleNamespace(list_ids=mock.AsyncMock(return_value=("s",))),
        mcp=types.SimpleNamespace(list_ids=mock.AsyncMock(return_value=("m",))),
    )


class TestLocalRuntimeClientMapping(unittest.TestCase):
    """LocalRuntimeClient maps protocol methods to runtime/storage/registries."""

    def test_cancel_delegates_to_runtime(self):
        bundle = _local_bundle()
        client = LocalRuntimeClient(bundle)
        _run(client.cancel("r1"))
        bundle.runtime.cancel.assert_awaited_once_with("r1")

    def test_approve_fences_then_resolves(self):
        request = types.SimpleNamespace(
            id="a1", status=ApprovalStatus.PENDING, version=7
        )
        bundle = _local_bundle(approval_request=request)
        client = LocalRuntimeClient(bundle)
        _run(client.approve("a1"))
        bundle.storage.approvals.approve.assert_awaited_once()
        _, kwargs = bundle.storage.approvals.approve.call_args
        self.assertEqual(kwargs["expected_version"], 7)

    def test_reject_fences_then_resolves(self):
        request = types.SimpleNamespace(
            id="a2", status=ApprovalStatus.PENDING, version=3
        )
        bundle = _local_bundle(approval_request=request)
        client = LocalRuntimeClient(bundle)
        _run(client.reject("a2", reason="no"))
        bundle.storage.approvals.reject.assert_awaited_once()
        _, kwargs = bundle.storage.approvals.reject.call_args
        self.assertEqual(kwargs["expected_version"], 3)
        self.assertEqual(kwargs["reason"], "no")

    def test_listers_hit_registries(self):
        bundle = _local_bundle()
        client = LocalRuntimeClient(bundle)
        self.assertEqual(_run(client.list_agents()), ("default", "reviewer"))
        self.assertEqual(_run(client.list_skills()), ("s",))
        self.assertEqual(_run(client.list_mcp_servers()), ("m",))

    def test_inspect_loads_spec_then_runtime_inspect(self):
        bundle = _local_bundle()
        client = LocalRuntimeClient(bundle)
        result = _run(client.inspect(None))
        bundle.agents.get.assert_awaited_once_with("default")
        bundle.runtime.inspect.assert_awaited_once_with("SPEC")
        self.assertEqual(result, "INSPECTION")

    def test_get_run_and_get_session_hit_storage(self):
        bundle = _local_bundle()
        client = LocalRuntimeClient(bundle)
        self.assertEqual(_run(client.get_run("r1")), "RUN")
        _run(client.get_session("main"))  # validates id, no raise

    def test_get_approval_reads_storage(self):
        request = types.SimpleNamespace(id="a1", tool_name="terminal")
        bundle = _local_bundle(approval_request=request)
        client = LocalRuntimeClient(bundle)
        self.assertIs(_run(client.get_approval("a1")), request)


if __name__ == "__main__":
    unittest.main()
