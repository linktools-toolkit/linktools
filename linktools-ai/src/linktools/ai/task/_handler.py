#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Application-owned TaskNode handler contracts."""

import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Generic, Literal, Protocol, TypeVar, runtime_checkable

from ..core import (
    ImmutableJsonMapping,
    JsonValue,
    Principal,
    CorrelationData,
    canonical_sha256,
    normalize_json_value,
    normalize_correlation,
)
from ._graph import TaskExpanderRef, TaskNode, TaskResultRef

AppT = TypeVar("AppT")
_TASK_TYPE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_RESERVED_TASK_TYPE_PREFIX = "linktools.ai."
_RESULT_DIGEST = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class TaskBindingSnapshot:
    """Durable semantic binding for one application-owned task handler."""

    task_type: str
    task_version: int
    effect: Literal["none", "replay_safe", "non_replay_safe"]
    output_contract: Mapping[str, JsonValue]
    timeout_seconds: float | None
    max_attempts: int
    retry_delay_seconds: float
    reconcile: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.task_type, str)
            or _TASK_TYPE.fullmatch(self.task_type) is None
            or self.task_type.startswith(_RESERVED_TASK_TYPE_PREFIX)
        ):
            raise ValueError("task binding type is invalid")
        if (
            not isinstance(self.task_version, int)
            or isinstance(self.task_version, bool)
            or self.task_version < 1
            or self.effect not in {"none", "replay_safe", "non_replay_safe"}
            or isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
            or isinstance(self.retry_delay_seconds, bool)
            or not isinstance(self.retry_delay_seconds, (int, float))
            or self.retry_delay_seconds < 0
            or (
                self.timeout_seconds is not None
                and (
                    isinstance(self.timeout_seconds, bool)
                    or not isinstance(self.timeout_seconds, (int, float))
                    or self.timeout_seconds <= 0
                )
            )
            or not isinstance(self.reconcile, bool)
        ):
            raise ValueError("task binding contract is invalid")
        normalized = normalize_json_value(dict(self.output_contract))
        if not isinstance(normalized, dict):
            raise ValueError("task output contract must be a JSON object")
        object.__setattr__(self, "output_contract", ImmutableJsonMapping(normalized))
        if self.timeout_seconds is not None:
            object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))
        object.__setattr__(self, "retry_delay_seconds", float(self.retry_delay_seconds))

    @property
    def binding_digest(self) -> str:
        return canonical_sha256(
            {
                "contract": "task-binding-v1",
                "task_type": self.task_type,
                "task_version": self.task_version,
                "effect": self.effect,
                "output_contract": dict(self.output_contract),
                "timeout_seconds": self.timeout_seconds,
                "max_attempts": self.max_attempts,
                "retry_delay_seconds": self.retry_delay_seconds,
                "reconcile": self.reconcile,
            }
        )

    @property
    def output_fingerprint(self) -> str:
        return canonical_sha256(
            {
                "contract": "task-output-v1",
                "output": dict(self.output_contract),
            }
        )


@dataclass(frozen=True, slots=True)
class TaskDependency:
    node_id: str
    output: JsonValue
    result_digest: str
    execution_id: "str | None" = None

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id.strip():
            raise ValueError("task dependency node id is required")
        if (
            not isinstance(self.result_digest, str)
            or _RESULT_DIGEST.fullmatch(self.result_digest) is None
        ):
            raise ValueError("task dependency result digest is invalid")
        output = normalize_json_value(self.output)
        if canonical_sha256(output) != self.result_digest:
            raise ValueError("task dependency result digest does not match output")
        if self.execution_id is not None and (
            not isinstance(self.execution_id, str) or not self.execution_id.strip()
        ):
            raise ValueError("task dependency execution id is invalid")
        object.__setattr__(self, "output", output)


class TaskArtifactPublisher(Protocol):
    async def publish(
        self,
        name: str,
        chunks: AsyncIterator[bytes],
        *,
        media_type: str,
        expected_size: int,
        expected_digest: str,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class TaskEffectResolution:
    """A read-only reconciliation result for an external task effect."""

    kind: str
    value: JsonValue | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"applied", "not_applied", "unknown"}:
            raise ValueError("task effect resolution kind is invalid")
        if self.kind != "applied" and self.value is not None:
            raise ValueError("only an applied effect can carry a value")
        object.__setattr__(self, "value", normalize_json_value(self.value))


@dataclass(frozen=True, slots=True)
class TaskNodeContext(Generic[AppT]):
    app: AppT
    principal: Principal
    graph_id: str
    node_id: str
    execution_id: str
    input: Mapping[str, JsonValue]
    dependencies: Mapping[str, TaskDependency]
    idempotency_key: str
    correlation: CorrelationData = field(default_factory=dict)
    artifacts: "TaskArtifactPublisher | None" = None

    def __post_init__(self) -> None:
        if not isinstance(self.graph_id, str) or not self.graph_id.strip():
            raise ValueError("task graph id is required")
        if not isinstance(self.node_id, str) or not self.node_id.strip():
            raise ValueError("task node id is required")
        if not isinstance(self.execution_id, str) or not self.execution_id.strip():
            raise ValueError("task execution id is required")
        if (
            not isinstance(self.idempotency_key, str)
            or not self.idempotency_key.strip()
        ):
            raise ValueError("task idempotency key is required")
        if not isinstance(self.input, Mapping):
            raise TypeError("task node input must be a mapping")
        normalized_input = normalize_json_value(dict(self.input))
        if not isinstance(normalized_input, dict):
            raise TypeError("task node input must be a mapping")
        dependencies = dict(self.dependencies)
        if any(
            not isinstance(key, str)
            or not isinstance(value, TaskDependency)
            for key, value in dependencies.items()
        ):
            raise ValueError("task dependency mapping is invalid")
        object.__setattr__(self, "input", ImmutableJsonMapping(normalized_input))
        object.__setattr__(self, "dependencies", MappingProxyType(dependencies))
        object.__setattr__(self, "correlation", normalize_correlation(self.correlation))

    def read_dependency(self, name: str) -> TaskDependency:
        """Read one dependency by its declared node identity."""
        if not isinstance(name, str) or not name:
            raise KeyError(name)
        try:
            return self.dependencies[name]
        except KeyError as error:
            raise KeyError(name) from error


@runtime_checkable
class TaskNodeHandler(Protocol[AppT]):
    @property
    def type(self) -> str: ...

    @property
    def version(self) -> int: ...

    def normalize(
        self,
        input: Mapping[str, JsonValue],
    ) -> Mapping[str, JsonValue]: ...

    async def run(self, context: TaskNodeContext[AppT]) -> JsonValue: ...

    async def cancel(self, context: TaskNodeContext[AppT]) -> None: ...


@dataclass(frozen=True)
class TaskFunction(Generic[AppT]):
    type: str
    version: int
    function: Callable[[TaskNodeContext[AppT]], Awaitable[JsonValue]] = field(
        repr=False,
        compare=False,
    )
    def __post_init__(self) -> None:
        if (
            not isinstance(self.type, str)
            or _TASK_TYPE.fullmatch(self.type) is None
            or self.type.startswith(_RESERVED_TASK_TYPE_PREFIX)
        ):
            raise ValueError("task handler type is invalid")
        if (
            not isinstance(self.version, int)
            or isinstance(self.version, bool)
            or self.version < 1
        ):
            raise ValueError("task handler version must be positive")
        if not callable(self.function):
            raise TypeError("task handler function must be callable")

    def normalize(
        self,
        input: Mapping[str, JsonValue],
    ) -> Mapping[str, JsonValue]:
        if not isinstance(input, Mapping):
            raise TypeError("task input must be a mapping")
        normalized = normalize_json_value(dict(input))
        if not isinstance(normalized, dict):
            raise TypeError("task input must be a mapping")
        if "type" in normalized or "version" in normalized:
            raise ValueError("task handler input cannot contain reserved fields")
        return normalized

    async def run(self, context: TaskNodeContext[AppT]) -> JsonValue:
        return await self.function(context)

    async def cancel(self, context: TaskNodeContext[AppT]) -> None:
        return None

    def node(
        self,
        node_id: str,
        *,
        input: "Mapping[str, JsonValue] | None" = None,
        dependencies: "tuple[str, ...]" = (),
        budget_cost: int = 1,
        expander: "TaskExpanderRef | None" = None,
        input_refs: "Mapping[str, TaskResultRef] | None" = None,
        timeout_seconds: "float | None" = None,
        max_attempts: int = 1,
        retry_delay_seconds: float = 0,
    ) -> TaskNode:
        normalized = self.normalize({} if input is None else input)
        return TaskNode(
            node_id,
            dependencies,
            input={
                "type": self.type,
                "version": self.version,
                **normalized,
            },
            budget_cost=budget_cost,
            expander=expander,
            input_refs=input_refs,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
        )


__all__ = [
    "TaskBindingSnapshot",
    "TaskDependency",
    "TaskArtifactPublisher",
    "TaskEffectResolution",
    "TaskFunction",
    "TaskNodeContext",
    "TaskNodeHandler",
]
