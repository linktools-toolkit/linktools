#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import _tool as tool_module
from linktools.ai.runtime._capabilities import ToolOperationDecision
from linktools.ai.runtime._tool import RuntimeToolOperationBridge
from linktools.ai.runtime.state._durability import DurableCommitResult, DurableCommitState
from linktools.ai.storage import InMemoryObjectStore, PayloadPolicy


class _Repository:
    pass


@pytest.mark.asyncio
async def test_committed_unknown_effect_raises_boundary_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def committed(*args: Any, **kwargs: Any) -> DurableCommitResult[object]:
        del args, kwargs
        return DurableCommitResult(DurableCommitState.COMMITTED)

    monkeypatch.setattr(tool_module, "run_durable_commit", committed)
    bridge = RuntimeToolOperationBridge(
        _Repository(),  # type: ignore[arg-type]
        InMemoryObjectStore(),
        namespace="runtime",
        tenant_id="tenant",
        execution_id="execution",
        step_run_id="step",
        binding_digest="b" * 64,
        owner="owner",
        background_tasks=set(),
        payload_policy=PayloadPolicy(),
    )

    with pytest.raises(AIError) as raised:
        await bridge.unknown(
            ToolOperationDecision("operation", "owner", 1, False),
            RuntimeError("unsafe payload must not escape"),
        )

    assert raised.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
    assert raised.value.safe_details == {
        "execution_id": "execution",
        "operation_id": "operation",
        "phase": "tool_effect",
    }
