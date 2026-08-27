#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for local Runtime launch and stream ownership."""

import asyncio
from types import SimpleNamespace

import pytest
from linktools.ai.core import (
    ExecutionDeltaType,
    ExecutionStatus,
    Page,
    Principal,
    ResourceKind,
    ResourceRef,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._coordinator import _LocalRuntimeCoordinator
from linktools.ai.runtime._event import DefaultEventService, ExecutionDelta, LiveExecutionEventBroker
from linktools.ai.runtime._execution import DefaultExecutionService
from linktools.ai.runtime.service_api import ExecutionStreamEvent


class _AllowAuthorization:
    async def authorize(self, principal: object, action: object, resource: object) -> None:
        del principal, action, resource


class _DenyAuthorization:
    async def authorize(self, principal: object, action: object, resource: object) -> None:
        del principal, action, resource
        raise AIError(ErrorCode.AUTHORIZATION_DENIED)


class _ExecutionRepository:
    async def get_header(self, execution_id: str, *, tenant_id: str) -> ResourceRef:
        return ResourceRef(ResourceKind.EXECUTION, execution_id, tenant_id)

    async def get(self, execution_id: str, *, tenant_id: str) -> object:
        del execution_id, tenant_id
        return SimpleNamespace(status=ExecutionStatus.STARTED, event_sequence=0)


class _EventRepository:
    async def list(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        after_sequence: int,
        limit: int,
    ) -> Page[object]:
        del execution_id, tenant_id, after_sequence, limit
        return Page(())


class _ExecutionService:
    async def request_terminal_handoff(self, execution_id: str, *, tenant_id: str) -> None:
        del execution_id, tenant_id


def _event_service(
    authorization: object,
    broker: LiveExecutionEventBroker,
) -> DefaultEventService:
    return DefaultEventService(
        _ExecutionRepository(),
        _EventRepository(),
        authorization,
        lambda execution_id, *, tenant_id: None,
        broker,
    )


def _coordinator(
    authorization: object,
) -> tuple[_LocalRuntimeCoordinator, LiveExecutionEventBroker]:
    broker = LiveExecutionEventBroker()
    coordinator = _LocalRuntimeCoordinator(
        _ExecutionService(),
        _event_service(authorization, broker),
    )
    return coordinator, broker


@pytest.mark.asyncio
async def test_prepared_reservation_is_one_shot_and_idempotent() -> None:
    broker = LiveExecutionEventBroker()
    broker.prepare_local_producer("execution")
    broker.prepare_local_producer("execution")
    broker.register_local_producer("execution", 0)

    live = broker.claim_local_producer("execution")
    assert live is not None
    assert broker.claim_local_producer("execution") is None

    broker.complete("execution")
    await live.close()
    assert not broker.is_local_producer("execution")


@pytest.mark.asyncio
async def test_non_stream_path_can_abandon_prepared_reservation() -> None:
    broker = LiveExecutionEventBroker()
    broker.prepare_local_producer("execution")
    broker.register_local_producer("execution", 0)

    broker.abandon_prepared_local_producer("execution")
    broker.complete("execution")

    assert not broker.is_local_producer("execution")


@pytest.mark.asyncio
async def test_stream_authorization_failure_does_not_consume_reservation() -> None:
    coordinator, broker = _coordinator(_DenyAuthorization())
    principal = Principal("principal", "tenant", "service")
    broker.prepare_local_producer("execution")
    broker.register_local_producer("execution", 0)

    denied = coordinator.stream("execution", principal=principal).__aiter__()
    with pytest.raises(AIError) as error:
        await denied.__anext__()
    assert error.value.code is ErrorCode.AUTHORIZATION_DENIED

    broker.publish(
        ExecutionDelta(
            "execution",
            ExecutionDeltaType.ASSISTANT_TEXT_DELTA,
            "still-buffered",
        )
    )
    allowed = _event_service(_AllowAuthorization(), broker)
    stream = allowed.stream("execution", principal=principal).__aiter__()
    first = await stream.__anext__()
    assert first.payload["text"] == "still-buffered"
    await stream.aclose()


@pytest.mark.asyncio
async def test_prepared_stream_replays_early_delta() -> None:
    coordinator, broker = _coordinator(_AllowAuthorization())
    principal = Principal("principal", "tenant", "service")
    broker.prepare_local_producer("execution")
    broker.register_local_producer("execution", 0)
    broker.publish(
        ExecutionDelta(
            "execution",
            ExecutionDeltaType.ASSISTANT_TEXT_DELTA,
            "early",
        )
    )

    stream = coordinator.stream("execution", principal=principal).__aiter__()
    first = await stream.__anext__()
    assert isinstance(first, ExecutionStreamEvent)
    assert first.event_type is ExecutionDeltaType.ASSISTANT_TEXT_DELTA
    assert first.payload["text"] == "early"
    await stream.aclose()


@pytest.mark.asyncio
async def test_stream_before_producer_registration_is_not_ready_without_consuming_reservation() -> None:
    broker = LiveExecutionEventBroker()
    service = _event_service(_AllowAuthorization(), broker)
    principal = Principal("principal", "tenant", "service")
    broker.prepare_local_producer("execution")

    early = service.stream("execution", principal=principal).__aiter__()
    with pytest.raises(AIError) as error:
        await early.__anext__()
    assert error.value.code is ErrorCode.EXECUTION_NOT_READY

    broker.register_local_producer("execution", 0)
    broker.publish(
        ExecutionDelta(
            "execution",
            ExecutionDeltaType.ASSISTANT_TEXT_DELTA,
            "registered",
        )
    )
    ready = service.stream("execution", principal=principal).__aiter__()
    first = await ready.__anext__()
    assert first.payload["text"] == "registered"
    await ready.aclose()


@pytest.mark.asyncio
async def test_abandon_after_claim_does_not_close_claimed_stream() -> None:
    broker = LiveExecutionEventBroker()
    service = _event_service(_AllowAuthorization(), broker)
    principal = Principal("principal", "tenant", "service")
    broker.prepare_local_producer("execution")
    broker.register_local_producer("execution", 0)
    broker.publish(
        ExecutionDelta("execution", ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "one")
    )

    stream = service.stream("execution", principal=principal).__aiter__()
    assert (await stream.__anext__()).payload["text"] == "one"

    broker.abandon_prepared_local_producer("execution")
    broker.publish(
        ExecutionDelta("execution", ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "two")
    )
    assert (await stream.__anext__()).payload["text"] == "two"
    await stream.aclose()


@pytest.mark.asyncio
async def test_second_stream_close_does_not_close_first_subscription() -> None:
    broker = LiveExecutionEventBroker()
    service = _event_service(_AllowAuthorization(), broker)
    principal = Principal("principal", "tenant", "service")
    broker.prepare_local_producer("execution")
    broker.register_local_producer("execution", 0)
    broker.publish(
        ExecutionDelta("execution", ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "initial")
    )

    first = service.stream("execution", principal=principal).__aiter__()
    assert (await first.__anext__()).payload["text"] == "initial"
    second = service.stream("execution", principal=principal).__aiter__()
    assert (await second.__anext__()).payload["text"] == "initial"
    await second.aclose()

    broker.publish(
        ExecutionDelta("execution", ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "next")
    )
    assert (await first.__anext__()).payload["text"] == "next"
    await first.aclose()


@pytest.mark.asyncio
async def test_wait_authorizes_before_abandoning_stream() -> None:
    service = object.__new__(DefaultExecutionService)
    abandoned: list[str] = []
    service._local_stream_abort = abandoned.append

    async def denied(
        execution_id: str,
        principal: Principal,
        action: object,
    ) -> object:
        del execution_id, principal, action
        raise AIError(ErrorCode.AUTHORIZATION_DENIED)

    service._load_authorized = denied
    principal = Principal("principal", "tenant", "service")
    with pytest.raises(AIError) as error:
        await DefaultExecutionService.wait.__wrapped__(
            service,
            "execution",
            principal=principal,
        )
    assert error.value.code is ErrorCode.AUTHORIZATION_DENIED
    assert abandoned == []


class _LaunchExecutions:
    async def get(self, execution_id: str, *, tenant_id: str) -> object:
        return SimpleNamespace(
            execution_id=execution_id,
            tenant_id=tenant_id,
            status=ExecutionStatus.STARTED,
        )


def _launch_service(backend: object) -> tuple[DefaultExecutionService, list[str], list[str]]:
    service = object.__new__(DefaultExecutionService)
    service._state = SimpleNamespace(executions=_LaunchExecutions())
    service._backend = backend
    prepared: list[str] = []
    abandoned: list[str] = []
    service._local_stream_prepare = prepared.append
    service._local_stream_abort = abandoned.append
    return service, prepared, abandoned


@pytest.mark.asyncio
async def test_launch_normal_return_without_worker_abandons_prepared_stream() -> None:
    class Backend:
        async def launch(self, request: object, execution: object) -> None:
            del request, execution

        def worker_installed(self, execution_id: str) -> bool:
            del execution_id
            return False

    service, prepared, abandoned = _launch_service(Backend())
    await service._launch_started(
        SimpleNamespace(),
        SimpleNamespace(execution_id="execution", tenant_id="tenant"),
        scope="execution.run",
        idempotency_key_digest="digest",
        prepare_local_stream=True,
    )
    assert prepared == ["execution"]
    assert abandoned == ["execution"]


@pytest.mark.asyncio
async def test_existing_worker_skips_new_prepared_reservation() -> None:
    class Backend:
        async def launch(self, request: object, execution: object) -> None:
            del request, execution

        def worker_installed(self, execution_id: str) -> bool:
            del execution_id
            return True

    service, prepared, abandoned = _launch_service(Backend())
    await service._launch_started(
        SimpleNamespace(),
        SimpleNamespace(execution_id="execution", tenant_id="tenant"),
        scope="execution.run",
        idempotency_key_digest="digest",
        prepare_local_stream=True,
    )
    assert prepared == []
    assert abandoned == []


@pytest.mark.asyncio
async def test_cancellation_before_worker_installation_abandons_reservation() -> None:
    class Backend:
        async def launch(self, request: object, execution: object) -> None:
            del request, execution
            raise asyncio.CancelledError

        def worker_installed(self, execution_id: str) -> bool:
            del execution_id
            return False

    service, prepared, abandoned = _launch_service(Backend())
    with pytest.raises(asyncio.CancelledError):
        await service._launch_started(
            SimpleNamespace(),
            SimpleNamespace(execution_id="execution", tenant_id="tenant"),
            scope="execution.run",
            idempotency_key_digest="digest",
            prepare_local_stream=True,
        )
    assert prepared == ["execution"]
    assert abandoned == ["execution"]


@pytest.mark.asyncio
async def test_cancellation_after_worker_installation_retains_reservation() -> None:
    class Backend:
        def __init__(self) -> None:
            self.installed = False

        async def launch(self, request: object, execution: object) -> None:
            del request, execution
            self.installed = True
            raise asyncio.CancelledError

        def worker_installed(self, execution_id: str) -> bool:
            del execution_id
            return self.installed

    service, prepared, abandoned = _launch_service(Backend())
    with pytest.raises(asyncio.CancelledError):
        await service._launch_started(
            SimpleNamespace(),
            SimpleNamespace(execution_id="execution", tenant_id="tenant"),
            scope="execution.run",
            idempotency_key_digest="digest",
            prepare_local_stream=True,
        )
    assert prepared == ["execution"]
    assert abandoned == []


@pytest.mark.asyncio
async def test_wait_timeout_includes_initial_authorized_read() -> None:
    service = object.__new__(DefaultExecutionService)
    service._local_stream_abort = None

    async def slow_authorized(
        execution_id: str,
        principal: Principal,
        action: object,
    ) -> object:
        del execution_id, principal, action
        await asyncio.sleep(0.05)
        return SimpleNamespace(execution_id="execution", status=ExecutionStatus.STARTED)

    service._load_authorized = slow_authorized
    principal = Principal("principal", "tenant", "service")
    with pytest.raises(AIError) as error:
        await DefaultExecutionService.wait.__wrapped__(
            service,
            "execution",
            principal=principal,
            timeout_seconds=0.001,
        )
    assert error.value.code is ErrorCode.EXECUTION_WAIT_TIMEOUT
