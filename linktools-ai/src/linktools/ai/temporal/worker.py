#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fail-closed production Worker registration."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from linktools.core import environ

from ..core.errors import ErrorCode, AIError
from .activity import EvaluationActivity, ExecuteActivity, SessionActivity, TaskActivity
from .workflow.suite import EvaluationWorkflow
from .workflow.run import ExecutionWorkflow
from .workflow.mutation import SessionWorkflow
from .workflow.dag import TaskWorkflow

if TYPE_CHECKING:
    from temporalio.api.common.v1 import Payload
    from temporalio.client import Client
    from temporalio.worker import Interceptor

try:
    from temporalio.api.common.v1 import Payload as _TemporalPayload
    from temporalio.converter import DataConverter as _TemporalDataConverter
    from temporalio.converter import PayloadCodec as _TemporalPayloadCodec
    from temporalio.worker import Interceptor as _TemporalInterceptor
    from temporalio.worker import Worker as _TemporalSdkWorker
except ModuleNotFoundError as error:
    if error.name != "temporalio":
        raise
    _TemporalPayload = None
    _TemporalDataConverter = None
    _TemporalPayloadCodec = None
    _TemporalInterceptor = None
    _TemporalSdkWorker = None

WorkflowType = type[ExecutionWorkflow] | type[SessionWorkflow] | type[TaskWorkflow] | type[EvaluationWorkflow]
ActivityType = ExecuteActivity | SessionActivity | TaskActivity | EvaluationActivity
DATA_CONVERTER = "json"
PAYLOAD_CODEC = "asset"
INTERCEPTOR = "linktools-ai"
BUILD_ID = "linktools-ai"
TASK_QUEUE = "linktools-ai-production"
_logger = environ.get_logger("ai.temporal.worker")


class TemporalWorker(Protocol):
    def register_workflows(self, workflows: 'Sequence[WorkflowType]') -> None: ...
    def register_activities(self, activities: 'Sequence[ActivityType]') -> None: ...
    def configure(
        self,
        *,
        data_converter: str,
        payload_codec: str,
        interceptor: str,
        build_id: str,
        task_queue: str,
    ) -> None: ...


class TemporalSdkClient(Protocol):
    def config(self, *, active_config: bool = False) -> 'TemporalSdkClientConfig': ...


class TemporalSdkClientConfig(Protocol):
    def __getitem__(self, key: str) -> 'TemporalSdkDataConverter': ...


class TemporalSdkDataConverter(Protocol):
    payload_codec: 'TemporalSdkPayloadCodec | None'


class TemporalSdkPayloadCodec(Protocol):
    async def encode(self, payloads: 'Sequence[Payload]') -> 'list[Payload]': ...

    async def decode(self, payloads: 'Sequence[Payload]') -> 'list[Payload]': ...


class TemporalSdkInterceptor(Protocol):
    pass


class TemporalSdkRuntime(Protocol):
    async def run(self) -> None: ...
    async def shutdown(self) -> None: ...


class TemporalActivity(Protocol):
    """Named decorated activity callable accepted by the SDK Worker."""

    __name__: str


_PAYLOAD_CODEC_METADATA_KEY = "linktools-ai-payload-codec"


if _TemporalPayloadCodec is not None:

    class AssetPayloadCodec(_TemporalPayloadCodec):
        """Tag Temporal payloads with the pinned Linktools AI codec revision."""

        async def encode(self, payloads: 'Sequence[Payload]') -> 'list[Payload]':
            return [_copy_payload(payload, marker=PAYLOAD_CODEC) for payload in payloads]

        async def decode(self, payloads: 'Sequence[Payload]') -> 'list[Payload]':
            decoded: list[Payload] = []
            for payload in payloads:
                marker = payload.metadata.get(_PAYLOAD_CODEC_METADATA_KEY)
                if marker and marker.decode("utf-8") != PAYLOAD_CODEC:
                    raise ValueError("unsupported Temporal payload codec")
                decoded.append(_copy_payload(payload, marker=None))
            return decoded

else:

    class AssetPayloadCodec:
        """Placeholder type used until the optional Temporal SDK is installed."""

        async def encode(self, payloads: 'Sequence[Payload]') -> 'list[Payload]':
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

        async def decode(self, payloads: 'Sequence[Payload]') -> 'list[Payload]':
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)


if _TemporalInterceptor is not None:

    class AssetWorkerInterceptor(_TemporalInterceptor):
        """Stable worker interceptor identity for the pinned production build."""

else:

    class AssetWorkerInterceptor:
        """Placeholder type used until the optional Temporal SDK is installed."""


def build_temporal_components() -> 'tuple[TemporalSdkDataConverter, TemporalSdkPayloadCodec, TemporalSdkInterceptor]':
    if _TemporalDataConverter is None or _TemporalPayloadCodec is None or _TemporalInterceptor is None:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "temporalio is required for production workers")
    payload_codec = AssetPayloadCodec()
    data_converter = _TemporalDataConverter(payload_codec=payload_codec)
    interceptor = AssetWorkerInterceptor()
    _logger.info("temporal components built: converter=%s codec=%s interceptor=%s", DATA_CONVERTER, PAYLOAD_CODEC, INTERCEPTOR)
    return data_converter, payload_codec, interceptor


def _copy_payload(payload: 'Payload', *, marker: 'str | None') -> 'Payload':
    if _TemporalPayload is None:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    copied = _TemporalPayload()
    copied.CopyFrom(payload)
    if marker is None:
        copied.metadata.pop(_PAYLOAD_CODEC_METADATA_KEY, None)
    else:
        copied.metadata[_PAYLOAD_CODEC_METADATA_KEY] = marker.encode("utf-8")
    return copied


@dataclass(frozen=True, slots=True)
class WorkerActivities:
    execution: ExecuteActivity
    session: SessionActivity
    task: TaskActivity
    evaluation: EvaluationActivity


@dataclass(frozen=True, slots=True)
class WorkerRegistration:
    workflows: "tuple[WorkflowType, ...]"
    activities: "tuple[ActivityType, ...]"
    data_converter: str = DATA_CONVERTER
    payload_codec: str = PAYLOAD_CODEC
    interceptor: str = INTERCEPTOR
    build_id: str = BUILD_ID
    task_queue: str = TASK_QUEUE

    def register(self, worker: TemporalWorker) -> None:
        expected_workflows = (ExecutionWorkflow, SessionWorkflow, TaskWorkflow, EvaluationWorkflow)
        expected_activities = (ExecuteActivity, SessionActivity, TaskActivity, EvaluationActivity)
        if self.workflows != expected_workflows or len(self.activities) != len(expected_activities):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if not all((self.data_converter, self.payload_codec, self.interceptor, self.build_id, self.task_queue)):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if not (
            isinstance(self.activities[0], expected_activities[0])
            and isinstance(self.activities[1], expected_activities[1])
            and isinstance(self.activities[2], expected_activities[2])
            and isinstance(self.activities[3], expected_activities[3])
        ):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if any(not _has_activity_options(activity) for activity in self.activities):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        try:
            worker.configure(
                data_converter=self.data_converter,
                payload_codec=self.payload_codec,
                interceptor=self.interceptor,
                build_id=self.build_id,
                task_queue=self.task_queue,
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY) from error
        worker.register_workflows(self.workflows)
        worker.register_activities(self.activities)
        _logger.info(
            "temporal worker registered: workflows=%s activities=%s build_id=%s task_queue=%s",
            len(self.workflows),
            len(self.activities),
            self.build_id,
            self.task_queue,
        )


class TemporalSdkWorker:
    """Validated adapter around the official Temporal SDK Worker."""

    def __init__(
        self,
        client: TemporalSdkClient,
        registration: WorkerRegistration,
        *,
        data_converter: TemporalSdkDataConverter,
        payload_codec: TemporalSdkPayloadCodec,
        interceptor: TemporalSdkInterceptor,
    ) -> None:
        if _TemporalSdkWorker is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "temporalio is required for production workers")
        try:
            if data_converter.payload_codec is not payload_codec:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "Temporal DataConverter must use the asset PayloadCodec")
            if client.config()["data_converter"] is not data_converter:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "Temporal client must use the asset DataConverter")
            self._registration = registration
            activities = _temporal_activity_functions(registration.activities)
            self._runtime = cast(
                TemporalSdkRuntime,
                _TemporalSdkWorker(
                    cast("Client", client),
                    task_queue=registration.task_queue,
                    activities=activities,
                    workflows=registration.workflows,
                    interceptors=(cast("Interceptor", interceptor),),
                    build_id=registration.build_id,
                    use_worker_versioning=True,
                ),
            )
        except AIError:
            raise
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY) from error

    def configure(
        self,
        *,
        data_converter: str,
        payload_codec: str,
        interceptor: str,
        build_id: str,
        task_queue: str,
    ) -> None:
        expected = (
            self._registration.data_converter,
            self._registration.payload_codec,
            self._registration.interceptor,
            self._registration.build_id,
            self._registration.task_queue,
        )
        if (data_converter, payload_codec, interceptor, build_id, task_queue) != expected:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

    def register_workflows(self, workflows: 'Sequence[WorkflowType]') -> None:
        if tuple(workflows) != self._registration.workflows:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

    def register_activities(self, activities: 'Sequence[ActivityType]') -> None:
        if tuple(activities) != self._registration.activities:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

    async def run(self) -> None:
        await self._runtime.run()

    async def shutdown(self) -> None:
        await self._runtime.shutdown()


def build_temporal_worker(
    client: TemporalSdkClient,
    registration: WorkerRegistration,
    *,
    data_converter: TemporalSdkDataConverter,
    payload_codec: TemporalSdkPayloadCodec,
    interceptor: TemporalSdkInterceptor,
) -> TemporalSdkWorker:
    worker = TemporalSdkWorker(
        client,
        registration,
        data_converter=data_converter,
        payload_codec=payload_codec,
        interceptor=interceptor,
    )
    registration.register(worker)
    return worker


def build_production_worker(
    client: TemporalSdkClient,
    activities: WorkerActivities,
) -> TemporalSdkWorker:
    registration = production_registration(activities)
    data_converter, payload_codec, interceptor = build_temporal_components()
    return build_temporal_worker(
        client,
        registration,
        data_converter=data_converter,
        payload_codec=payload_codec,
        interceptor=interceptor,
    )


def production_registration(activities: WorkerActivities) -> WorkerRegistration:
    return WorkerRegistration(
        (ExecutionWorkflow, SessionWorkflow, TaskWorkflow, EvaluationWorkflow),
        (activities.execution, activities.session, activities.task, activities.evaluation),
    )


def _has_activity_options(activity: ActivityType) -> bool:
    options = activity.options
    return all(
        isinstance(value, int) and value > 0
        for value in (
            options.start_to_close_seconds,
            options.retry_max_attempts,
            options.heartbeat_timeout_seconds,
        )
    )


def _temporal_activity_functions(activities: Sequence[ActivityType]) -> tuple[TemporalActivity, ...]:
    functions: list[TemporalActivity] = []
    for activity in activities:
        if isinstance(activity, ExecuteActivity):
            functions.extend(
                (
                    cast(TemporalActivity, activity.run),
                    cast(TemporalActivity, activity.load_input),
                    cast(TemporalActivity, activity.fix_bundle_route),
                    cast(TemporalActivity, activity.fix_binding),
                    cast(TemporalActivity, activity.load_prompt),
                    cast(TemporalActivity, activity.reserve_budget),
                    cast(TemporalActivity, activity.run_agent),
                    cast(TemporalActivity, activity.process_deferred),
                    cast(TemporalActivity, activity.commit_result),
                    cast(TemporalActivity, activity.settle_budget),
                )
            )
        elif isinstance(activity, (SessionActivity, TaskActivity, EvaluationActivity)):
            functions.append(cast(TemporalActivity, activity.run))
        else:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    return tuple(functions)


__all__ = [
    "ActivityType",
    "AssetPayloadCodec",
    "AssetWorkerInterceptor",
    "BUILD_ID",
    "DATA_CONVERTER",
    "INTERCEPTOR",
    "PAYLOAD_CODEC",
    "TASK_QUEUE",
    "TemporalWorker",
    "TemporalSdkClient",
    "TemporalSdkClientConfig",
    "TemporalSdkDataConverter",
    "TemporalSdkInterceptor",
    "TemporalSdkPayloadCodec",
    "TemporalSdkWorker",
    "build_temporal_components",
    "build_production_worker",
    "build_temporal_worker",
    "WorkerActivities",
    "WorkerRegistration",
    "WorkflowType",
    "production_registration",
]
