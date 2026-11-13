#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Terminal publication depends on durable facts, not a live model directory."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.capability import ExecutionCapability, TraceCapability
from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence
from linktools.ai.persistence.codec import decode_envelope, encode_envelope
from linktools.ai.runtime import Runtime, RuntimeFeatures

from ._runtime_process_test_helpers import _NAMESPACE, _RuntimeProcess, _storage
from ._runtime_test_helpers import native_checkpoint


def _features_without_model() -> RuntimeFeatures:
    features = RuntimeFeatures(_NAMESPACE)
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.capability(native_checkpoint(), scope="runtime")
    features.capability(TraceCapability(), scope="runtime")
    return features


@pytest.mark.asyncio
@pytest.mark.parametrize("recovery", ("startup", "explicit"))
async def test_terminal_handoff_survives_removed_model_route(
    tmp_path: Path, recovery: str,
) -> None:
    database = tmp_path / "runtime.db"
    owner = _RuntimeProcess(database, phase="finalizing")
    try:
        initial = owner.initial
        assert initial["status"] == "FINALIZING"
        assert initial["handoff_phase"] == "prepared"
        assert initial["model_calls"] == 1
        execution_id = initial["execution_id"]
        owner.crash()
        async with Persistence.open(_NAMESPACE, storage=_storage(database, "finalizing")) as persistence:
            async with Runtime.attach(
                persistence=persistence, features=_features_without_model(),
                auto_recover=recovery == "startup",
            ) as runtime:
                principal = runtime.default_principal
                if recovery == "explicit":
                    with pytest.raises(AIError) as denied:
                        await runtime.executions.recover(
                            execution_id, principal=replace(principal, principal_id="other-owner"),
                        )
                    assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
                    await runtime.executions.recover(execution_id, principal=principal)
                result = await runtime.executions.result(execution_id, principal=principal)
                assert result.status is ExecutionStatus.SUCCEEDED
                assert result.output == {"text": "owner answer"}
                record = await persistence.execution.executions.get(execution_id, tenant_id=principal.tenant_id)
                assert record.recovery.handoff_phase.value == "completed"
                assert record.producer_generation == initial["generation"]
                session = await persistence.conversation.sessions.get("session", tenant_id=principal.tenant_id)
                assert session.active_execution_id is None
                timeline = await runtime.sessions.timeline("session", principal=principal)
                assert len(timeline.items) == 1
                assert timeline.items[0].output == result.output
    finally:
        owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("recovery", ("startup", "explicit"))
@pytest.mark.parametrize("damage", ("binding", "outcome"))
async def test_terminal_handoff_rejects_corruption_without_model_route(
    tmp_path: Path, recovery: str, damage: str,
) -> None:
    database = tmp_path / "runtime.db"
    owner = _RuntimeProcess(database, phase="finalizing")
    try:
        execution_id = owner.initial["execution_id"]
        owner.crash()
        async with Persistence.open(_NAMESPACE, storage=_storage(database, "finalizing")) as persistence:
            repository = persistence.execution.executions

            async def corrupt(transaction):
                stored = await transaction.get_record(repository.execution_record_key(execution_id))
                assert stored is not None
                value = deepcopy(decode_envelope(stored.data).value)
                fields = value["payload"]["fields"]
                if damage == "binding":
                    fields["binding"]["output_mode"] = "invalid"
                else:
                    output = fields["recovery"]["fields"]["terminal_handoff"]["fields"]["output"]
                    content = output["fields"]["content"]
                    if "$native_text" in content:
                        content["$native_text"]["slices"][0]["first"] = "0" * 64
                    else:
                        content["$execution_payload"] = "execution-payload:" + "0" * 64
                malformed = replace(
                    stored, data=encode_envelope(value),
                    storage_revision=stored.storage_revision + 1,
                )
                assert await transaction.replace_record(
                    malformed, expected_storage_revision=stored.storage_revision,
                )

            await repository.state_store.mutate(corrupt)
            with pytest.raises(AIError) as rejected:
                async with Runtime.attach(
                    persistence=persistence, features=_features_without_model(),
                    auto_recover=recovery == "startup",
                ) as runtime:
                    await runtime.executions.recover(execution_id, principal=runtime.default_principal)
            assert rejected.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        owner.close()


@pytest.mark.asyncio
async def test_active_recovery_still_requires_model_route(tmp_path: Path) -> None:
    database = tmp_path / "runtime.db"
    owner = _RuntimeProcess(database, phase="active")
    try:
        execution_id = owner.initial["execution_id"]
        owner.crash()
        async with Persistence.open(_NAMESPACE, storage=_storage(database, "active")) as persistence:
            async with Runtime.attach(
                persistence=persistence, features=_features_without_model(), auto_recover=False,
            ) as runtime:
                principal = runtime.default_principal
                with pytest.raises(AIError) as unavailable:
                    await runtime.executions.recover(execution_id, principal=principal)
                assert unavailable.value.code is ErrorCode.AGENT_BINDING_UNAVAILABLE
                record = await persistence.execution.executions.get(execution_id, tenant_id=principal.tenant_id)
                assert record.producer_generation == owner.initial["generation"]
                assert record.recovery.handoff_phase.value == "none"
    finally:
        owner.close()
