#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from inspect import Parameter, signature

import pytest
from pydantic_ai.messages import BinaryContent

from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.core import ExecutionLineageKind, Principal
from linktools.ai.runtime import ExecutionRequest
from linktools.ai.runtime._execution import _request_digest
from linktools.ai.runtime._input import input_intent
from linktools.ai.runtime.state import StoredUserInput
from linktools.ai.runtime.state._contracts import RecoveryExecutionInput
from linktools.ai.spec import AgentSpec
from linktools.ai.storage import StoredPayload


def _binding() -> AgentBindingSnapshot:
    return AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("agent", model="model"),
        model={"route_id": "model", "model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode="text",
        output_schema={},
        binding_digest="a" * 64,
    )


def _request(prompt: str) -> ExecutionRequest:
    return ExecutionRequest(
        user_prompt=prompt,
        principal=Principal("user", "tenant", "service"),
        idempotency_key="same-key",
        memory_scope=None,
        mode="run",
        planning=False,
        thinking=False,
    )


def _execution_request_digest(request: ExecutionRequest) -> str:
    return _request_digest(
        request,
        _binding().binding_digest,
        session_id=None,
        source_execution_id=None,
        base_execution_id=None,
        parent_execution_id=None,
        root_execution_id=None,
        parent_invocation_id=None,
        lineage_kind=ExecutionLineageKind.RUN,
    )


def test_input_intent_ignores_file_body_and_keeps_logical_paths() -> None:
    first = input_intent("Inspect", ("evidence.txt",))
    second = input_intent("Inspect", ("evidence.txt",))

    assert first == second
    assert first.files == ("evidence.txt",)
    assert len(first.digest) == 64


def test_stored_user_input_has_one_versioned_owner() -> None:
    stored = StoredUserInput(
        1,
        "text",
        StoredPayload.inline_text("prompt"),
    )

    assert stored.digest == StoredUserInput(1, "text", stored.payload).digest


def test_recovery_input_requires_storage_contract() -> None:
    parameter = signature(RecoveryExecutionInput).parameters["storage_contract"]
    assert parameter.default is Parameter.empty


def test_text_request_digest_is_stable() -> None:
    first = _request('{"message":{"kind":"request"}}')
    second = _request('{"message":{"kind":"request"}}')

    assert _execution_request_digest(first) == _execution_request_digest(second)


def test_binary_input_intent_contains_metadata_without_body() -> None:
    value = (BinaryContent(b"body", media_type="text/plain", identifier="a.txt"),)

    intent = input_intent(value, ())

    assert intent.prompt[0]["kind"] == "binary"
    assert intent.prompt[0]["size"] == 4
    assert intent.prompt[0]["media_type"] == "text/plain"
    assert intent.prompt[0]["identifier"] == "a.txt"


def test_stored_user_input_does_not_accept_unknown_codec() -> None:
    with pytest.raises(ValueError):
        StoredUserInput(1, "legacy", StoredPayload.inline_text("prompt"))
