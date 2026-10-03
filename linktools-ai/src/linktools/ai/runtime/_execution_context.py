#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable, immutable framework context accepted before an Agent invocation."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pydantic_ai.messages import ModelMessage

from ..core import ImmutableJsonMapping, JsonValue, canonical_json_bytes, canonical_sha256
from ..errors import AIError, ErrorCode
from ._message import decode_model_messages, encode_model_messages


@dataclass(frozen=True, slots=True)
class ExecutionInputContext:
    history: bytes
    session_metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    memory: Mapping[str, JsonValue] | None = None
    repository_instructions: Mapping[str, JsonValue] | None = None
    replace_history_system_prompt: bool = False
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        decode_model_messages(self.history)
        for name in ("session_metadata", "memory", "repository_instructions"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, ImmutableJsonMapping(value))
        if self.unavailable_reason is not None and (not isinstance(self.unavailable_reason, str) or not self.unavailable_reason):
            raise ValueError("context unavailability reason is invalid")
        if not isinstance(self.replace_history_system_prompt, bool):
            raise ValueError("context history replacement flag is invalid")

    @classmethod
    def from_messages(cls, history: Sequence[ModelMessage], **kwargs: object) -> "ExecutionInputContext":
        return cls(encode_model_messages(history), **kwargs)

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_payload())

    def model_messages(self) -> tuple[ModelMessage, ...]:
        return tuple(decode_model_messages(self.history))

    def to_payload(self) -> dict[str, JsonValue]:
        return {"version": 1, "history": json.loads(self.history),
                "session_metadata": dict(self.session_metadata),
                "memory": None if self.memory is None else dict(self.memory),
                "repository_instructions": None if self.repository_instructions is None else dict(self.repository_instructions),
                "replace_history_system_prompt": self.replace_history_system_prompt,
                "unavailable_reason": self.unavailable_reason}

    @classmethod
    def from_payload(cls, value: Mapping[str, JsonValue]) -> "ExecutionInputContext":
        if value.get("version") != 1:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        try:
            return cls(canonical_json_bytes(value["history"]), value["session_metadata"], value["memory"],
                       value["repository_instructions"], value["replace_history_system_prompt"], value["unavailable_reason"])
        except (KeyError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


__all__ = ["ExecutionInputContext"]
