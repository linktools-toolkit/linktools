#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Typed, JSON-backed input values for Agent Task nodes."""

from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING

from ..core import (
    ImmutableJsonMapping,
    JsonValue,
    Principal,
    ThinkingValue,
    canonical_sha256,
    normalize_json_value,
    normalize_thinking,
    validate_memory_scope,
)
from ..errors import AIError, ErrorCode
from ..task import TaskDependencyState, TaskNodeInvocation, TaskResultRef
from ._input import decode_task_prompt_draft, task_prompt_draft, validate_user_input
from ._input_contract import CanonicalUserInput, UserPromptInput

if TYPE_CHECKING:
    from .state._contracts import StoredUserInput


class _AgentTaskContextError(AIError):
    """Marks an error owned by declared result access rather than user code."""


class AgentTaskInput(Mapping[str, JsonValue]):
    """Immutable input intent for a Runtime-owned Agent Task."""

    __slots__ = ("_values", "_stored_prompt")

    def __init__(
        self,
        prompt: UserPromptInput = "",
        *,
        parameters: Mapping[str, JsonValue] | None = None,
        files: Sequence[str] = (),
        session_id: str | None = None,
        memory_scope: str | None = None,
        planning: bool | None = None,
        thinking: ThinkingValue | None = None,
    ) -> None:
        try:
            canonical_prompt = validate_user_input(prompt)
            parameter_values = normalize_json_value(dict(parameters or {}))
            normalized_files = tuple(files)
            normalized_scope = (
                None
                if memory_scope is None
                else validate_memory_scope(memory_scope)
            )
            normalized_thinking = (
                None if thinking is None else normalize_thinking(thinking)
            )
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
        if (
            not isinstance(parameter_values, dict)
            or any(not isinstance(key, str) or not key for key in parameter_values)
            or any(not isinstance(path, str) or not path for path in normalized_files)
            or (session_id is not None and (not isinstance(session_id, str) or not session_id))
            or (planning is not None and not isinstance(planning, bool))
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        values: dict[str, JsonValue] = {
            "kind": "agent-task-input",
            "version": 1,
            "prompt": task_prompt_draft(canonical_prompt),
            "parameters": parameter_values,
            "files": list(normalized_files),
            "session_id": session_id,
            "memory_scope": normalized_scope,
            "planning": planning,
            "thinking": normalized_thinking,
        }
        self._values = ImmutableJsonMapping(values)
        self._stored_prompt: "StoredUserInput | None" = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, JsonValue]) -> "AgentTaskInput":
        if not isinstance(value, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if "version" not in value:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        version = value["version"]
        if isinstance(version, bool) or not isinstance(version, int):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if version != 1:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        required = {
            "kind",
            "version",
            "prompt",
            "parameters",
            "files",
            "session_id",
            "memory_scope",
            "planning",
            "thinking",
        }
        if not required.issubset(value) or value.get("kind") != "agent-task-input":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        prompt = value["prompt"]
        parameters = value["parameters"]
        files = value["files"]
        planning = value["planning"]
        thinking = value["thinking"]
        if (
            not isinstance(prompt, Mapping)
            or not isinstance(parameters, Mapping)
            or not isinstance(files, list)
            or not isinstance(planning, bool)
            or not isinstance(thinking, (bool, str))
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            stored_prompt = None
            if prompt.get("kind") == "stored-user-content-v1":
                source_intent_digest = prompt.get("source_intent_digest")
                if (
                    set(prompt)
                    != {
                        "kind",
                        "intent",
                        "source_intent_digest",
                        "value",
                    }
                    or prompt.get("intent") != "task-admission"
                    or not isinstance(source_intent_digest, str)
                    or len(source_intent_digest) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in source_intent_digest
                    )
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                from .state._codec import decode_domain
                from .state._contracts import StoredUserInput

                stored_prompt = decode_domain(prompt.get("value"), StoredUserInput)
                decoded_prompt = "Stored user input"
            else:
                decoded_prompt = decode_task_prompt_draft(prompt)
            result = cls(
                decoded_prompt,
                parameters=parameters,
                files=files,
                session_id=value["session_id"],
                memory_scope=value["memory_scope"],
                planning=planning,
                thinking=thinking,
            )
            normalized_values = dict(result._values)
            if stored_prompt is not None:
                normalized_values["prompt"] = dict(prompt)
                object.__setattr__(result, "_stored_prompt", stored_prompt)
            for key, item in value.items():
                if key not in required:
                    normalized_values[key] = normalize_json_value(item)
            object.__setattr__(result, "_values", ImmutableJsonMapping(normalized_values))
            return result
        except AIError as error:
            if error.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED:
                raise
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    @classmethod
    def from_authoring(
        cls,
        value: Mapping[str, JsonValue] | None,
    ) -> "AgentTaskInput":
        if value is None or (isinstance(value, Mapping) and not value):
            return cls()
        if not isinstance(value, Mapping):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        declared_fields = {
            "kind",
            "version",
            "prompt",
            "parameters",
            "files",
            "session_id",
            "memory_scope",
            "planning",
            "thinking",
        }
        if ("kind" in value or "version" in value) and not declared_fields.issubset(value):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if "kind" in value and value.get("kind") != "agent-task-input":
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if "version" in value:
            version = value.get("version")
            if (
                isinstance(version, bool)
                or not isinstance(version, int)
                or version != 1
            ):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        try:
            prompt_value = value.get("prompt", "")
            prompt = (
                decode_task_prompt_draft(prompt_value)
                if isinstance(prompt_value, Mapping)
                else prompt_value
            )
            result = cls(
                prompt,
                parameters=value.get("parameters", {}),
                files=value.get("files", ()),
                session_id=value.get("session_id"),
                memory_scope=value.get("memory_scope"),
                planning=value.get("planning"),
                thinking=value.get("thinking"),
            )
            normalized_values = dict(result._values)
            for key, item in value.items():
                if key not in {
                    "kind",
                    "version",
                    "prompt",
                    "parameters",
                    "files",
                    "session_id",
                    "memory_scope",
                    "planning",
                    "thinking",
                }:
                    normalized_values[key] = normalize_json_value(item)
            object.__setattr__(result, "_values", ImmutableJsonMapping(normalized_values))
            return result
        except AIError as error:
            if error.code is ErrorCode.REQUEST_FIELD_INVALID:
                raise
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error

    def execution_payload(self) -> Mapping[str, JsonValue]:
        return ImmutableJsonMapping(
            {
                key: self._values[key]
                for key in (
                    "kind",
                    "version",
                    "prompt",
                    "parameters",
                    "files",
                    "session_id",
                    "memory_scope",
                    "planning",
                    "thinking",
                )
            }
        )

    @property
    def prompt(self) -> CanonicalUserInput:
        if self._stored_prompt is not None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        prompt = self._values["prompt"]
        if not isinstance(prompt, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return decode_task_prompt_draft(prompt)

    @property
    def parameters(self) -> Mapping[str, JsonValue]:
        value = self._values["parameters"]
        if not isinstance(value, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return value

    @property
    def files(self) -> tuple[str, ...]:
        value = self._values["files"]
        if not isinstance(value, list):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return tuple(value)

    @property
    def session_id(self) -> str | None:
        value = self._values["session_id"]
        return value if isinstance(value, str) else None

    @property
    def memory_scope(self) -> str | None:
        value = self._values["memory_scope"]
        return value if isinstance(value, str) else None

    @property
    def planning(self) -> bool | None:
        value = self._values["planning"]
        return value if isinstance(value, bool) else None

    @property
    def thinking(self) -> ThinkingValue | None:
        value = self._values["thinking"]
        return normalize_thinking(value) if isinstance(value, (bool, str)) else None

    @property
    def stored_prompt(self) -> "StoredUserInput | None":
        return self._stored_prompt

    def __getitem__(self, key: str) -> JsonValue:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


class AgentTaskInputContext:
    """Read-only model input projection context with declared result access."""

    __slots__ = (
        "_invocation",
        "_prompt",
        "_result_reader",
        "_result_ref_reader",
        "_input",
        "_result_cache",
        "_result_ref_cache",
    )

    def __init__(
        self,
        invocation: TaskNodeInvocation,
        task_input: AgentTaskInput,
        result_reader: Callable[[str], Awaitable[JsonValue]],
        result_ref_reader: Callable[[str], Awaitable[TaskResultRef]],
    ) -> None:
        self._invocation = invocation
        self._prompt = task_input.prompt
        self._result_reader = result_reader
        self._result_ref_reader = result_ref_reader
        self._input = ImmutableJsonMapping(task_input.parameters)
        self._result_cache: dict[str, JsonValue] = {}
        self._result_ref_cache: dict[str, TaskResultRef] = {}

    @property
    def graph_id(self) -> str:
        return self._invocation.graph_id

    @property
    def node_id(self) -> str:
        return self._invocation.node.node_id

    @property
    def principal(self) -> Principal:
        return self._invocation.principal

    @property
    def input(self) -> Mapping[str, JsonValue]:
        return self._input

    @property
    def prompt(self) -> CanonicalUserInput:
        return self._prompt

    @property
    def dependency_states(self) -> Mapping[str, TaskDependencyState]:
        return MappingProxyType(dict(self._invocation.dependency_states))

    @property
    def source_refs(self) -> tuple[tuple[str, TaskResultRef], ...]:
        return tuple(sorted(self._result_ref_cache.items()))

    async def result(self, name: str) -> JsonValue:
        selected = self._validate_name(name)
        if selected not in self._result_ref_cache:
            await self.result_ref(selected)
        if selected not in self._result_cache:
            try:
                value = await self._result_reader(selected)
            except AIError as error:
                raise _AgentTaskContextError(
                    error.code,
                    safe_details=error.safe_details,
                ) from error
            reference = self._result_ref_cache[selected]
            if canonical_sha256(value) != reference.result_digest:
                raise _AgentTaskContextError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._result_cache[selected] = value
        return normalize_json_value(self._result_cache[selected])

    async def result_ref(self, name: str) -> TaskResultRef:
        selected = self._validate_name(name)
        if selected not in self._result_ref_cache:
            try:
                self._result_ref_cache[selected] = await self._result_ref_reader(selected)
            except AIError as error:
                raise _AgentTaskContextError(
                    error.code,
                    safe_details=error.safe_details,
                ) from error
        return self._result_ref_cache[selected]

    def _validate_name(self, name: str) -> str:
        if not isinstance(name, str) or not name:
            raise _AgentTaskContextError(ErrorCode.REQUEST_FIELD_INVALID)
        if name not in self._invocation.node.dependencies and name not in self._invocation.node.input_refs:
            raise _AgentTaskContextError(ErrorCode.REQUEST_FIELD_INVALID)
        return name


AgentTaskInputBuilder = Callable[
    [AgentTaskInputContext], Awaitable[UserPromptInput]
]


__all__ = ["AgentTaskInput", "AgentTaskInputBuilder", "AgentTaskInputContext"]
