#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Named task definitions and exact references."""

import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

from pydantic import BaseModel

from ..core import ImmutableJsonMapping, JsonValue, Principal, normalize_json_value
from ..errors import AIError
from ..spec import canonicalize_json_schema, canonicalize_pydantic_model_schema

if TYPE_CHECKING:
    from ._graph import TaskNode
    from ._handler import TaskEffectResolution, TaskNodeContext
    from ._runner import TaskNodeRunner

AppT = TypeVar("AppT")
_TASK_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_TASK_TYPE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")


@dataclass(frozen=True, slots=True)
class TaskRef:
    id: str
    revision: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.id, str)
            or _TASK_ID.fullmatch(self.id) is None
            or isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 1
        ):
            raise ValueError("task reference is invalid")

    @classmethod
    def deferred_input(cls) -> "TaskRef":
        return cls("linktools.ai.input", 1)


@dataclass(frozen=True, slots=True)
class TaskExpanderRef:
    id: str
    revision: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.id, str)
            or _TASK_ID.fullmatch(self.id) is None
            or isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 1
        ):
            raise ValueError("task expander reference is invalid")


class TaskExpansionContext(Protocol):
    @property
    def principal(self) -> Principal: ...

    @property
    def graph_id(self) -> str: ...

    @property
    def source_node(self) -> "TaskNode": ...

    @property
    def output(self) -> JsonValue: ...


@dataclass(frozen=True, slots=True, init=False)
class TaskExpander:
    id: str
    revision: int
    expand: Callable[[TaskExpansionContext], Sequence["TaskNode"]] = field(
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        id: str,
        expand: Callable[[TaskExpansionContext], Sequence["TaskNode"]],
        *,
        revision: int = 1,
    ) -> None:
        TaskExpanderRef(id, revision)
        if id.startswith("linktools.ai."):
            raise ValueError("task expander id is reserved")
        if not callable(expand):
            raise TypeError("task expander callback must be callable")
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "expand", expand)

    @property
    def ref(self) -> TaskExpanderRef:
        return TaskExpanderRef(self.id, self.revision)


class Task(Generic[AppT]):
    """One named execution definition; graph nodes retain only its reference."""

    __slots__ = (
        "_ref",
        "_function",
        "_runner",
        "_normalizer",
        "_cancel_callback",
        "_reconcile_callback",
        "_output_type",
        "_contract",
    )

    def __init__(
        self,
        id: str,
        run: Callable[["TaskNodeContext[AppT]"], Awaitable[JsonValue]],
        *,
        revision: int = 1,
        effect_policy: str = "non_replay_safe",
        output_type: type[BaseModel] | None = None,
        normalize: Callable[
            [Mapping[str, JsonValue]], Mapping[str, JsonValue]
        ] | None = None,
        cancel: Callable[["TaskNodeContext[AppT]"], Awaitable[None]] | None = None,
        reconcile: Callable[
            ["TaskNodeContext[AppT]"], Awaitable["TaskEffectResolution"]
        ] | None = None,
    ) -> None:
        reference = _definition_ref(id, revision)
        if not callable(run):
            raise TypeError("task run callback must be callable")
        if effect_policy not in {"none", "replay_safe", "non_replay_safe"}:
            raise ValueError("task effect policy is invalid")
        if normalize is not None and not callable(normalize):
            raise TypeError("task normalize callback must be callable")
        if cancel is not None and not callable(cancel):
            raise TypeError("task cancel callback must be callable")
        if reconcile is not None and not callable(reconcile):
            raise TypeError("task reconcile callback must be callable")
        output_contract = _output_contract(output_type)
        contract: dict[str, JsonValue] = {
            "version": 1,
            "type": "function",
            "effect_policy": effect_policy,
            "output_contract": output_contract,
            "reconcile": reconcile is not None,
        }
        self._initialize(
            reference,
            run,
            None,
            _normalize_input if normalize is None else normalize,
            cancel,
            reconcile,
            output_type,
            contract,
        )

    @classmethod
    def from_runner(
        cls,
        id: str,
        runner: "TaskNodeRunner[AppT]",
        *,
        revision: int = 1,
        contract: Mapping[str, JsonValue],
    ) -> "Task[AppT]":
        reference = _definition_ref(id, revision)
        if runner is None:
            raise TypeError("task runner is required")
        declaration = _normalize_runner_contract(contract)
        value = cls.__new__(cls)
        value._initialize(
            reference,
            None,
            runner,
            _normalize_input,
            None,
            None,
            None,
            declaration,
        )
        return value

    def _initialize(
        self,
        reference: TaskRef,
        function: Callable[["TaskNodeContext[AppT]"], Awaitable[JsonValue]] | None,
        runner: "TaskNodeRunner[AppT] | None",
        normalizer: Callable[
            [Mapping[str, JsonValue]], Mapping[str, JsonValue]
        ],
        cancel_callback: Callable[["TaskNodeContext[AppT]"], Awaitable[None]] | None,
        reconcile_callback: Callable[
            ["TaskNodeContext[AppT]"], Awaitable["TaskEffectResolution"]
        ] | None,
        output_type: type[BaseModel] | None,
        contract: Mapping[str, JsonValue],
    ) -> None:
        if (function is None) == (runner is None):
            raise ValueError("task must have exactly one function or runner")
        try:
            normalized_contract = ImmutableJsonMapping(contract)
        except (TypeError, ValueError) as error:
            raise ValueError("task declaration is invalid") from error
        object.__setattr__(self, "_ref", reference)
        object.__setattr__(self, "_function", function)
        object.__setattr__(self, "_runner", runner)
        object.__setattr__(self, "_normalizer", normalizer)
        object.__setattr__(self, "_cancel_callback", cancel_callback)
        object.__setattr__(self, "_reconcile_callback", reconcile_callback)
        object.__setattr__(self, "_output_type", output_type)
        object.__setattr__(self, "_contract", normalized_contract)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Task definitions are immutable")

    @property
    def id(self) -> str:
        return self._ref.id

    @property
    def revision(self) -> int:
        return self._ref.revision

    @property
    def ref(self) -> TaskRef:
        return self._ref

    @property
    def function(
        self,
    ) -> Callable[["TaskNodeContext[AppT]"], Awaitable[JsonValue]] | None:
        return self._function

    @property
    def runner(self) -> "TaskNodeRunner[AppT] | None":
        return self._runner

    @property
    def normalizer(
        self,
    ) -> Callable[[Mapping[str, JsonValue]], Mapping[str, JsonValue]]:
        return self._normalizer

    @property
    def cancel_callback(
        self,
    ) -> Callable[["TaskNodeContext[AppT]"], Awaitable[None]] | None:
        return self._cancel_callback

    @property
    def reconcile_callback(
        self,
    ) -> Callable[["TaskNodeContext[AppT]"], Awaitable["TaskEffectResolution"]] | None:
        return self._reconcile_callback

    @property
    def output_type(self) -> type[BaseModel] | None:
        return self._output_type

    @property
    def contract(self) -> Mapping[str, JsonValue]:
        return self._contract

    @property
    def effect_policy(self) -> str:
        value = self._contract["effect_policy"]
        assert isinstance(value, str)
        return value

    def normalize(self, value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        normalized = self._normalizer(value)
        if not isinstance(normalized, Mapping):
            raise TypeError("task normalizer must return a mapping")
        payload = normalize_json_value(dict(normalized))
        if not isinstance(payload, dict):
            raise TypeError("task normalizer must return a mapping")
        return ImmutableJsonMapping(payload)


def _definition_ref(id: str, revision: int) -> TaskRef:
    reference = TaskRef(id, revision)
    if id.startswith("linktools.ai."):
        raise ValueError("task id is reserved")
    return reference


def _normalize_input(value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise TypeError("task input must be a mapping")
    normalized = normalize_json_value(dict(value))
    if not isinstance(normalized, dict):
        raise TypeError("task input must be a mapping")
    return normalized


def _output_contract(output_type: type[BaseModel] | None) -> dict[str, JsonValue]:
    if output_type is None:
        return {"kind": "json"}
    if not isinstance(output_type, type) or not issubclass(output_type, BaseModel):
        raise TypeError("task output_type must be a Pydantic model")
    return {
        "kind": "schema",
        "schema": canonicalize_pydantic_model_schema(output_type),
    }


def _normalize_runner_contract(
    contract: Mapping[str, JsonValue],
) -> Mapping[str, JsonValue]:
    if not isinstance(contract, Mapping):
        raise TypeError("runner task contract must be a mapping")
    value = normalize_json_value(dict(contract))
    if not isinstance(value, dict):
        raise ValueError("runner task contract is invalid")
    required = {"version", "type", "effect_policy", "output_contract", "reconcile"}
    allowed = required | {"config"}
    if not required.issubset(value):
        raise ValueError("runner task contract is incomplete")
    if set(value) - allowed:
        raise ValueError("runner task contract is invalid")
    if (
        value["version"] != 1
        or isinstance(value["version"], bool)
        or not isinstance(value["type"], str)
        or _TASK_TYPE.fullmatch(value["type"]) is None
        or value["type"] == "function"
        or value["effect_policy"] not in {"none", "replay_safe", "non_replay_safe"}
        or not isinstance(value["reconcile"], bool)
        or not isinstance(value["output_contract"], Mapping)
    ):
        raise ValueError("runner task contract is invalid")
    output = value["output_contract"]
    if output.get("kind") == "json":
        if set(output) != {"kind"}:
            raise ValueError("runner output contract is invalid")
    elif output.get("kind") == "schema":
        schema = output.get("schema")
        if set(output) != {"kind", "schema"} or not isinstance(schema, Mapping):
            raise ValueError("runner output contract is invalid")
        try:
            canonical_schema = canonicalize_json_schema(schema)
        except AIError as error:
            raise ValueError("runner output contract is invalid") from error
        value["output_contract"] = {
            "kind": "schema",
            "schema": canonical_schema,
        }
    else:
        raise ValueError("runner output contract is invalid")
    if "config" in value and not isinstance(value["config"], Mapping):
        raise ValueError("runner task config is invalid")
    return ImmutableJsonMapping(value)


__all__ = [
    "Task",
    "TaskExpansionContext",
    "TaskExpander",
    "TaskExpanderRef",
    "TaskRef",
]
