#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent requests keep their admitted content identity across codec upgrades."""

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest
from pydantic_ai.messages import StructuredContent, TextContent

from linktools.ai.core import (ExecutionLineageKind, ExecutionStatus, IdempotencyStatus, Principal, PromptLimits, ResourceKind, WorkspaceFileInput,
                               canonical_json_bytes, idempotency_key_digest)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ExecutionRecord, ExecutionRequest, ExecutionStartClaim, IdempotencyRecord, StoredUserInput
from linktools.ai.inputs import ExecutionInputMaterializer, encode_user_content
from linktools.ai.model import ModelBinding, create_model
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.codec import decode_domain, encode_domain
from linktools.ai.runtime._execution import ExecutionStartIdentity, _request_digest
from linktools.ai.storage import StoredPayload

from .persistence.test_execution_start_service import _Launcher, _binding, _request, _service
from .test_history_projection_conformance import _record
from .test_mixed_pending_admission import _start, _writers
from . import test_mixed_pending_admission as pending_fixture
from .test_pending_coordination import _ObservedCache, _claim, _helper
from .test_session_admission_conformance import _session

_PRINCIPAL = Principal("owner", "tenant", "service")


def _structured(value: int = 1, tag: str = "state") -> StructuredContent:
    return StructuredContent({"count": value}, content="fixed rendering", tag=tag)


def _digest(request: ExecutionRequest, binding_digest: str) -> str:
    return _request_digest(
        request, binding_digest, session_id=None, previous_execution_id=None,
        fork_base_execution_id=None, parent_execution_id=None, root_execution_id=None,
        parent_invocation_id=None, lineage_kind=ExecutionLineageKind.RUN,
    )


class _ClaimingLauncher(_Launcher):
    async def prepare_start(
        self, request: ExecutionRequest, execution: ExecutionRecord, identity: ExecutionStartIdentity,
    ) -> ExecutionRecord:
        return await self._repository.claim_start(ExecutionStartClaim(
            execution.execution_id, execution.revision, execution.event_seq,
            identity.scope, identity.idempotency_key_digest, identity.request_digest,
            execution.created_at, identity.request_identity_version,
        ))


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", (_structured(2), _structured(tag="other")))
async def test_new_agent_requests_bind_complete_structured_content(changed: StructuredContent) -> None:
    async with Persistence.open("structured-agent", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
        backend = _ClaimingLauncher(state.execution.executions)
        service = _service(state, backend=backend)
        request = replace(_request("placeholder", _PRINCIPAL, "same"), user_prompt=(_structured(),))
        binding = _binding()
        first = await service.start(binding.binding_digest, request)
        assert await service.start(binding.binding_digest, request) == first
        assert await service.resolve_existing(binding.binding_digest, request) == first
        for action in (service.start, service.resolve_existing):
            with pytest.raises(AIError) as caught:
                await action(binding.binding_digest, replace(request, user_prompt=(changed,)))
            assert caught.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        identity = await state.execution.idempotency.get(
            "execution.run", idempotency_key_digest("same"), tenant_id="tenant",
        )
        assert identity.request_identity_version == 2
        assert backend.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("storage_kind", ("memory", "sqlite"))
@pytest.mark.parametrize("session", (False, True))
async def test_old_agent_identity_replays_without_rewriting_owner(storage_kind: str, session: bool, tmp_path: Path) -> None:
    storage = (PersistenceStorage.in_memory() if storage_kind == "memory" else
               PersistenceStorage.from_root(tmp_path, exclusive_writer=True))
    async with Persistence.open("legacy-agent", storage=storage, tenant_id="tenant") as state:
        backend = _ClaimingLauncher(state.execution.executions)
        service = _service(state, backend=backend)
        old_request = replace(_request("placeholder", _PRINCIPAL, "same"), user_prompt=(TextContent("fixed rendering"),))
        binding = _binding()
        scope = "session.resume" if session else "execution.run"
        if session:
            await state.conversation.sessions.create(replace(_session(), agent_id="agent"))

        async def start(value):
            return await service._start(
                binding.binding_digest, value, scope=scope,
                session_id="session" if session else None,
                session_agent_id="agent" if session else None,
            )

        first = await start(old_request)
        owner = await state.execution.idempotency.get(
            scope, idempotency_key_digest("same"), tenant_id="tenant",
        )
        wire = encode_domain(owner)
        del wire["fields"]["request_identity_version"]
        legacy = decode_domain(wire, IdempotencyRecord)
        assert legacy.request_identity_version == 1
        await state.execution.idempotency.compare_and_swap(
            legacy.scope, legacy.idempotency_key_digest, tenant_id="tenant",
            expected_status=owner.status, next_record=legacy,
        )
        request = replace(old_request, user_prompt=(_structured(),))
        assert await start(request) == first
        if not session:
            assert await service.resolve_existing(binding.binding_digest, request) == first
        assert await state.execution.idempotency.get(
            legacy.scope, legacy.idempotency_key_digest, tenant_id="tenant",
        ) == legacy
        assert backend.calls == 1
        with pytest.raises(AIError) as caught:
            await start(replace(request, user_prompt=(StructuredContent(
                {"count": 1}, content="changed rendering", tag="state",
            ),)))
        assert caught.value.code is ErrorCode.IDEMPOTENCY_CONFLICT


@pytest.mark.asyncio
async def test_agent_rechecks_concurrent_legacy_owner_using_original_request(monkeypatch: pytest.MonkeyPatch) -> None:
    async with Persistence.open("identity-race", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
        backend = _ClaimingLauncher(state.execution.executions)
        service = _service(state, backend=backend)
        binding = _binding()
        request = replace(_request("placeholder", _PRINCIPAL, "same"), user_prompt=(_structured(),))
        old_request = replace(request, user_prompt=(TextContent("fixed rendering"),))
        reserve = state.execution.executions.reserve_start
        inserted = False

        async def race(proposed, **kwargs):
            nonlocal inserted
            if not inserted:
                inserted = True
                legacy = replace(
                    proposed,
                    execution=replace(proposed.execution, execution_id="legacy-winner", root_execution_id="legacy-winner",
                                      stored_user_input=StoredUserInput("user-content-v1", StoredPayload.inline_json(
                                          encode_user_content(old_request.user_prompt)))),
                    idempotency=replace(proposed.idempotency, resource_id="legacy-winner",
                                        request_digest=_digest(old_request, binding.binding_digest), request_identity_version=1),
                )
                await reserve(legacy)
            return await reserve(proposed, **kwargs)

        monkeypatch.setattr(state.execution.executions, "reserve_start", race)
        result = await service.start(binding.binding_digest, request)
        assert result.execution_id == "legacy-winner"
        assert backend.calls == 1
        owner = await state.execution.idempotency.get("execution.run", idempotency_key_digest("same"), tenant_id="tenant")
        assert owner.request_identity_version == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("with_file", (False, True))
async def test_local_and_remote_pending_retries_use_the_retained_owner_version(
    with_file: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    if with_file:
        def file_model(*, api: str, model: str) -> ModelBinding:
            return create_model(api=api, model=model, input_types=("mime:text/plain",))
        monkeypatch.setattr(pending_fixture, "create_model", file_model)

    class OnceReadableInput:
        def __init__(self) -> None:
            self.reads = 0

        async def canonicalize_path(self, path: str) -> str:
            return path

        async def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes:
            self.reads += 1
            if self.reads > 1:
                raise FileNotFoundError(path)
            return b"accepted input"

        async def close(self) -> None:
            pass

    sources = (OnceReadableInput(), OnceReadableInput())
    async with _writers() as (_state, pending, remote):
        if with_file:
            for service, source in zip((pending, remote), sources):
                service._input_materializer = ExecutionInputMaterializer(source, PromptLimits())
        async def start(request, reservation, session, coordination):
            reservation = replace(reservation, idempotency=replace(reservation.idempotency, request_identity_version=1))
            turn, created = pending._runtime_bridge.pending_turns.admit("tenant", reservation, session,
                                                                        now=reservation.execution.created_at)
            assert created
            await pending._runtime_bridge.pending_turns.coordination.publish(
                turn.execution, coordination, request_identity_version=1,
            )
            return turn.execution.execution_id

        pending._runtime_bridge.pending_start = start
        attachments = (WorkspaceFileInput("input.txt"),) if with_file else ()
        first = await _start(pending, prompt=(TextContent("fixed rendering"), *attachments))
        assert await _start(pending, prompt=(_structured(), *attachments)) == first
        assert await _start(remote, prompt=(_structured(), *attachments)) == first
        assert remote.runtime_backend().calls == 0
        assert [source.reads for source in sources] == ([1, 1] if with_file else [0, 0])


@pytest.mark.asyncio
@pytest.mark.parametrize("version", (1, 2))
async def test_pending_identity_version_requires_confirmed_owner_and_retains_old_omission(version: int) -> None:
    cache = _ObservedCache()
    helper = _helper(cache)
    owned = await _claim(helper)
    await helper.publish(_record(ExecutionStatus.STARTED, 1), owned, request_identity_version=version)
    if version == 1:
        key = cache.record_key()
        value = json.loads(await cache.get(key))
        del value["request_identity_version"]
        await cache.put(key, canonical_json_bytes(value))
    assert await helper.read_request_identity_version("request", tenant_id="tenant") == version
    await cache.delete(cache.record_key())
    with pytest.raises(AIError) as caught:
        await helper.read_request_identity_version("request", tenant_id="tenant")
    assert caught.value.code is ErrorCode.EXECUTION_START_UNKNOWN
    await cache.release_turn(owned.lease)
    assert await helper.read_request_identity_version("request", tenant_id="tenant") is None


@pytest.mark.asyncio
async def test_legacy_failure_receipt_selects_identity_without_execution_payload() -> None:
    async with Persistence.open("legacy-failure", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
        backend = _ClaimingLauncher(state.execution.executions)
        service = _service(state, backend=backend)
        binding = _binding()
        request = replace(_request("placeholder", _PRINCIPAL, "same"), user_prompt=(_structured(),))
        old_request = replace(request, user_prompt=(TextContent("fixed rendering"),))
        now = datetime.now(timezone.utc)
        owner = IdempotencyRecord(
            "execution.run", idempotency_key_digest("same"), _digest(old_request, binding.binding_digest),
            ResourceKind.EXECUTION, "missing-execution", IdempotencyStatus.FAILED, None,
            ErrorCode.EXECUTION_START_PERSISTENCE_FAILED.value, now, now,
        )
        await state.execution.idempotency.reserve(owner)
        with pytest.raises(AIError) as caught:
            await service.start(binding.binding_digest, request)
        assert caught.value.code is ErrorCode.EXECUTION_START_PERSISTENCE_FAILED
        assert backend.calls == 0
