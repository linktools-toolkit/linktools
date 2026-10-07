#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime composition and ownership invariants."""

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from linktools.ai.agent import AgentBindingContract
from linktools.ai.agent._output import bind_output, restore_output
from linktools.ai.asset import AssetKey, AssetStore, InMemoryAssetBackend
from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import (
    ExecutionLineageKind,
    ExecutionStatus,
    Principal,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import Runtime
from linktools.ai.runtime import _factory as runtime_factory
from linktools.ai.runtime._factory import compose_runtime_components
from linktools.ai.runtime._subagent import SubagentDispatcher
from linktools.ai.runtime.state import RuntimeStorage
from linktools.ai.runtime.state._codec import decode_domain, encode_domain
from linktools.ai.runtime.state._contracts import ExecutionRecord, StoredUserInput
from linktools.ai.spec import AgentSpec
from linktools.ai.storage import StorageOverlay, StoredPayload
from pydantic import BaseModel

from ._runtime_test_helpers import RuntimeUsageModels


class _UncertainExecution:
    async def cancel(self, execution_id: str, request: object) -> object:
        del execution_id, request
        return SimpleNamespace(cancelled=False)

    async def inspect(self, execution_id: str, *, principal: Principal) -> object:
        del execution_id, principal
        return SimpleNamespace(status=ExecutionStatus.STARTED)


def _binding() -> AgentBindingContract:
    output = bind_output()
    return AgentBindingContract(
        agent_spec=AgentSpec("agent", model="model"),
        model_contract={"route_id": "model", "model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode=output.mode,
        output_schema=output.schema_definition,
    )


def _execution(*, binding: AgentBindingContract | None = None) -> ExecutionRecord:
    now = datetime.now(timezone.utc)
    selected = binding or _binding()
    return ExecutionRecord(
        execution_id="execution",
        session_id=None,
        parent_execution_id=None,
        root_execution_id="execution",
        previous_execution_id=None,
        fork_base_execution_id=None,
        lineage_kind=ExecutionLineageKind.RUN,
        status=ExecutionStatus.PENDING_START,
        revision=0,
        event_sequence=0,
        agent_run_sequence=0,
        error_code=None,
        safe_error_details={},
        created_at=now,
        updated_at=now,
        mode="run",
        planning=False,
        thinking=False,
        binding=selected,
        principal_id="principal",
        principal_kind="service",
        stored_user_input=StoredUserInput(
            "text",
            StoredPayload.inline_text("prompt"),
        ),
    )


@pytest.mark.asyncio
async def test_runtime_does_not_close_borrowed_asset_store(tmp_path: Path) -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend))
    await store.initialize()
    group = CapabilityGroup("workspace", assets=store)
    components = await compose_runtime_components(
        "workspace",
        models=ModelRegistry.openai(model="gpt-test"),
        storage=RuntimeStorage.in_memory(),
        capabilities=(group,),
    )

    await components.close_callback()

    assert store.ready
    await store.close()
    await backend.close()


@pytest.mark.asyncio
async def test_runtime_rejects_source_change_during_assembly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    original_build = runtime_factory._build_local_components

    async def build_then_change_source(**kwargs: object):
        components = await original_build(**kwargs)  # type: ignore[arg-type]
        await store.put(AssetKey("agent", "late"), b"changed")
        return components

    monkeypatch.setattr(
        runtime_factory,
        "_build_local_components",
        build_then_change_source,
    )
    try:
        with pytest.raises(AIError) as error:
            await compose_runtime_components(
                "workspace",
                models=RuntimeUsageModels(),  # type: ignore[arg-type]
                storage=RuntimeStorage.in_memory(),
                capabilities=(CapabilityGroup("workspace", assets=store),),
            )

        assert error.value.code is ErrorCode.SNAPSHOT_CONFLICT
        assert store.ready
    finally:
        await store.close()
        await backend.close()


def test_output_contract_restores_only_mode_and_schema() -> None:
    class LocalOutput(BaseModel):
        value: str

    automatic = bind_output(LocalOutput)
    restored = restore_output(automatic.mode, automatic.schema_definition)

    assert automatic.mode == "structured"
    assert restored.mode == automatic.mode
    assert restored.schema_definition == automatic.schema_definition

    with pytest.raises(AIError) as restore_error:
        restore_output("structured", {"type": "not-a-json-schema-type"})
    assert restore_error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID


@pytest.mark.parametrize(
    ("factory", "target"),
    ((_execution, ExecutionRecord),),
)
def test_binding_codec_round_trips_mandatory_exact_v1_shape(
    factory: object,
    target: type[object],
) -> None:
    current = factory(binding=_binding())
    assert decode_domain(encode_domain(current), target) == current


@pytest.mark.parametrize(
    ("factory", "target"),
    ((_execution, ExecutionRecord),),
)
def test_binding_codec_rejects_partial_current_v1_shapes(
    factory: object,
    target: type[object],
) -> None:
    wire = encode_domain(factory(binding=_binding()))
    assert isinstance(wire, dict)
    fields = dict(wire["fields"])
    fields.pop("binding")
    partial = dict(wire)
    partial["fields"] = fields

    with pytest.raises(AIError) as error:
        decode_domain(partial, target)

    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_subagent_unknown_cancel_requires_recovery() -> None:
    dispatcher = object.__new__(SubagentDispatcher)
    dispatcher._execution = _UncertainExecution()

    with pytest.raises(AIError) as error:
        await dispatcher.cancel_child(
            "execution",
            parent_execution_id="parent",
            principal=Principal("principal", "tenant", "service"),
        )

    assert error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED


@pytest.mark.asyncio
async def test_runtime_persists_model_usage_through_history_views() -> None:
    application = CapabilityGroup("application")
    application.agent("default", model="default", allow_tools=())

    async with Runtime.open(
        "default",
        models=RuntimeUsageModels(),  # type: ignore[arg-type]
        storage=RuntimeStorage.in_memory(),
        capabilities=(application,),
    ) as runtime:
        result = (await runtime.agents.get("default").run(
            "hello",
            timeout_seconds=10,
        )).result
        assert result.status is ExecutionStatus.SUCCEEDED

        history = await runtime.executions.history(
            result.execution_id,
            principal=runtime.default_principal,
        )
        transcript = await runtime.executions.transcript(
            result.execution_id,
            principal=runtime.default_principal,
        )
        trace = await runtime.executions.trace(
            result.execution_id,
            principal=runtime.default_principal,
        )

    responses = [
        item
        for item in trace.items
        if item.payload.get("kind") == "MODEL_RESPONSE"
    ]
    assert len(responses) == 1
    assert responses[0].payload["token_usage"] == {
        "input_tokens": 101,
        "output_tokens": 202,
        "cache_read_tokens": 303,
        "cache_write_tokens": 404,
    }
    assert history.items
    assert transcript.items
