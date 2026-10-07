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
from linktools.ai.runtime._execution import (
    DefaultExecutionService,
    _ExecutionRuntimeBridge,
)
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
        return SimpleNamespace(status=ExecutionStatus.STARTED, event_seq=0)


class _EventRepository:
    async def list(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        after_event_seq: int,
        limit: int,
    ) -> Page[object]:
        del execution_id, tenant_id, after_event_seq, limit
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
    service._runtime_bridge = _ExecutionRuntimeBridge()

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


class _LaunchExecutions:
    tenant_id = "tenant"

    async def get(self, execution_id: str, *, tenant_id: str) -> object:
        return SimpleNamespace(
            execution_id=execution_id,
            tenant_id=tenant_id,
            status=ExecutionStatus.STARTED,
        )


def _launch_service(backend: object) -> tuple[DefaultExecutionService, list[str], list[str]]:
    service = object.__new__(DefaultExecutionService)
    service._state = SimpleNamespace(executions=_LaunchExecutions())
    bridge = _ExecutionRuntimeBridge()
    bridge.bind(backend)  # type: ignore[arg-type]
    service._runtime_bridge = bridge
    prepared: list[str] = []
    abandoned: list[str] = []

    class Broker:
        def prepare_local_producer(self, execution_id: str) -> None:
            prepared.append(execution_id)

        def abandon_prepared_local_producer(self, execution_id: str) -> None:
            abandoned.append(execution_id)

    service._live_broker = Broker()
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
    service._runtime_bridge = _ExecutionRuntimeBridge()

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


@pytest.mark.asyncio
@pytest.mark.parametrize("denied_at", ("header", "event", "execution", None))
async def test_event_list_authorizes_before_reading_and_preserves_page(denied_at: str | None) -> None:
    from unittest.mock import AsyncMock

    from linktools.ai.core import AuthorizationAction
    from linktools.ai.runtime import ExecutionEvent

    principal = Principal("principal", "tenant", "service")
    header = ResourceRef(ResourceKind.EXECUTION, "execution", "tenant")
    actions: list[AuthorizationAction] = []
    error = AIError(ErrorCode.AUTHORIZATION_DENIED)

    async def authorize(actor: Principal, action: AuthorizationAction, resource: ResourceRef) -> None:
        assert actor is principal
        assert resource is header
        actions.append(action)
        if (denied_at == "event" and action is AuthorizationAction.EVENT_READ) or (
            denied_at == "execution" and action is AuthorizationAction.EXECUTION_READ
        ):
            raise error

    executions = SimpleNamespace(get_header=AsyncMock(return_value=None if denied_at == "header" else header))
    events = SimpleNamespace(list=AsyncMock(return_value=Page(
        (ExecutionEvent("execution", 7, "EXECUTION_STARTED", {"value": "stored"}),),
        "next-page",
    )))
    service = DefaultEventService(
        executions, events, SimpleNamespace(authorize=authorize), lambda *args, **kwargs: None
    )
    if denied_at is not None:
        with pytest.raises(AIError) as caught:
            await service.list("execution", principal=principal, after_event_seq=6, limit=2)
        assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED
        if denied_at != "header":
            assert caught.value is error
        events.list.assert_not_awaited()
    else:
        page = await service.list("execution", principal=principal, after_event_seq=6, limit=2)
        assert page == events.list.return_value
        events.list.assert_awaited_once_with("execution", tenant_id="tenant", after_event_seq=6, limit=2)
    assert actions == {
        "header": [],
        "event": [AuthorizationAction.EVENT_READ],
        "execution": [AuthorizationAction.EVENT_READ, AuthorizationAction.EXECUTION_READ],
        None: [AuthorizationAction.EVENT_READ, AuthorizationAction.EXECUTION_READ],
    }[denied_at]
    executions.get_header.assert_awaited_once_with("execution", tenant_id="tenant")


@pytest.mark.asyncio
async def test_event_list_preserves_repository_failure() -> None:
    from unittest.mock import AsyncMock

    failure = RuntimeError("durable read failed")
    events = SimpleNamespace(list=AsyncMock(side_effect=failure))
    service = DefaultEventService(
        _ExecutionRepository(), events, _AllowAuthorization(), lambda *args, **kwargs: None
    )
    with pytest.raises(RuntimeError) as caught:
        await service.list("execution", principal=Principal("principal", "tenant", "service"))
    assert caught.value is failure


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["claim_local_producer", "is_local_producer", "base_event_seq", "subscribe"])
@pytest.mark.parametrize("typed", [False, True])
async def test_event_broker_preserves_typed_failures(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    typed: bool,
) -> None:
    from linktools.ai.runtime.service_api import _ExecutionStreamFailure

    cause = AIError(ErrorCode.STORAGE_UNAVAILABLE) if typed else OSError("broker failed")
    broker = LiveExecutionEventBroker()
    broker.register_local_producer("execution", 0)

    def fail(*args, **kwargs):
        raise cause

    monkeypatch.setattr(broker, operation, fail)
    service = _event_service(_AllowAuthorization(), broker)
    with pytest.raises(AIError if typed else _ExecutionStreamFailure) as raised:
        await anext(service.stream("execution", principal=Principal("owner", "tenant")))

    assert (raised.value if typed else raised.value.cause) is cause


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["iterate", "close", "activity"])
@pytest.mark.parametrize("typed", [False, True])
async def test_live_subscription_preserves_typed_failures(operation: str, typed: bool) -> None:
    from linktools.ai.runtime._event import _close_live, _iterate_live, _wait_live_activity
    from linktools.ai.runtime.service_api import _ExecutionStreamFailure

    cause = AIError(ErrorCode.STORAGE_UNAVAILABLE) if typed else OSError("broker failed")

    class Subscription:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise cause

        async def close(self):
            raise cause

        async def wait_for_activity(self, _execution_id):
            raise cause

    subscription = Subscription()
    with pytest.raises(AIError if typed else _ExecutionStreamFailure) as raised:
        if operation == "iterate":
            await anext(_iterate_live(subscription))
        elif operation == "close":
            await _close_live(subscription)
        else:
            await _wait_live_activity(subscription, "execution", timeout=1)

    assert (raised.value if typed else raised.value.cause) is cause


@pytest.mark.asyncio
async def test_live_stream_missing_base_sequence_is_authoritative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = LiveExecutionEventBroker()
    broker.register_local_producer("execution", 0)
    monkeypatch.setattr(broker, "base_event_seq", lambda _execution_id: None)
    service = _event_service(_AllowAuthorization(), broker)
    with pytest.raises(AIError) as raised:
        await anext(service.stream("execution", principal=Principal("owner", "tenant")))

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_live_close_authoritative_error_overrides_optional_broker_error():
    cause = AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    class Live:
        def is_local_producer(self, execution_id):
            return True
        def base_event_seq(self, execution_id):
            return 0
        def subscribe(self, execution_id):
            return self
        def __aiter__(self):
            return self
        async def __anext__(self):
            raise RuntimeError("broker unavailable")
        async def close(self):
            raise cause
    service = _event_service(_AllowAuthorization(), Live())
    with pytest.raises(AIError) as raised:
        await anext(service._stream_with_live("execution", principal=Principal("owner", "tenant"),
                                             after_event_seq=0, authorized=True))
    assert raised.value is cause


@pytest.mark.asyncio
async def test_event_stream_close_awaits_owned_live_subscription():
    from linktools.ai.core import ExecutionDeltaType
    from linktools.ai.runtime._event import ExecutionDelta
    class Live:
        closed = False
        def claim_local_producer(self, execution_id):
            return self
        def is_local_producer(self, execution_id):
            return True
        def base_event_seq(self, execution_id):
            return 0
        def __aiter__(self):
            return self
        async def __anext__(self):
            return ExecutionDelta("execution", ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "text")
        async def close(self):
            self.closed = True
    live = Live()
    service = _event_service(_AllowAuthorization(), live)
    stream = service.stream("execution", principal=Principal("owner", "tenant"))
    await anext(stream)
    await stream.aclose()
    assert live.closed
