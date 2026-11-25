#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolved Agent Task nodes own prompt-intent identity independently of storage."""

from collections.abc import Mapping
from dataclasses import dataclass
import re

from ..core import JsonValue, canonical_sha256
from ..errors import AIError, ErrorCode
from ._graph import TaskNode


@dataclass(frozen=True, init=False)
class AgentTaskNode(TaskNode):
    """An Agent invocation resolved by Runtime, with one authoritative input."""

    def prompt_intent_digest(self) -> str:
        """Retain the canonical prompt identity when its bytes are materialized."""
        body = self.input
        prompt = body.get("prompt")
        if (
            body.get("kind") != "agent-task-input"
            or type(body.get("version")) is not int
            or body["version"] != 1
            or not isinstance(prompt, Mapping)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if prompt.get("kind") == "stored-user-content-v1":
            intent_digest = prompt.get("source_intent_digest")
            if (
                set(prompt) != {"kind", "intent", "source_intent_digest", "value"}
                or prompt.get("intent") != "task-admission"
                or not isinstance(intent_digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", intent_digest) is None
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return intent_digest
        return canonical_sha256(dict(prompt))

    def to_mapping(self) -> dict[str, JsonValue]:
        value = super().to_mapping()
        if self.task is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if self.input_capture is None:
            value["input"] = {
                **self.input,
                "prompt": {"kind": "task-prompt-intent-v1", "digest": self.prompt_intent_digest()},
            }
        return value


__all__ = ["AgentTaskNode"]
