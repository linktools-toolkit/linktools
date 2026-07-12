#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/events/test_payloads_run_model_tool.py"""

from linktools.ai.events.payloads import (
    ModelCompleted,
    ModelFailed,
    ModelStarted,
    RunCancelled,
    RunCompleted,
    RunFailed,
    RunPaused,
    RunResumed,
    RunStarted,
    ToolCompleted,
    ToolFailed,
    ToolStarted,
)


def test_run_lifecycle_payloads():
    assert RunStarted(run_id="r1", runnable_id="a1").run_id == "r1"
    assert dict(RunCompleted(run_id="r1").result_summary) == {}
    assert RunFailed(run_id="r1", error_type="X", message="boom").message == "boom"
    assert RunPaused(run_id="r1").reason is None
    assert RunResumed(run_id="r1").run_id == "r1"
    assert RunCancelled(run_id="r1").reason is None


def test_model_lifecycle_payloads():
    assert ModelStarted(model_type="gpt-4").model_type == "gpt-4"
    assert dict(ModelCompleted(model_type="gpt-4").token_usage) == {}
    assert (
        ModelFailed(model_type="gpt-4", error_message="timeout").error_message
        == "timeout"
    )


def test_tool_lifecycle_payloads():
    assert ToolStarted(tool_name="file", tool_call_id="call-1").tool_name == "file"
    assert (
        ToolCompleted(tool_name="file", tool_call_id="call-1", success=True).success
        is True
    )
    assert (
        ToolFailed(
            tool_name="file", tool_call_id="call-1", error_message="denied"
        ).error_message
        == "denied"
    )
