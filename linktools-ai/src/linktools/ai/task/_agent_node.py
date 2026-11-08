#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolved Agent Task nodes own prompt-intent identity independently of storage."""

from collections.abc import Mapping
from dataclasses import dataclass
import re

from ..core import JsonValue, canonical_sha256
from ..errors import AIError, ErrorCode
from ..model.messages import project_user_content_identity
from ._graph import TaskNode


@dataclass(frozen=True, init=False)
class AgentTaskNode(TaskNode):
    """An Agent invocation resolved by Runtime, with one authoritative input."""

    @classmethod
    def prompt_identity_payload(
        cls, prompt: Mapping[str, JsonValue], *, request_identity_version: int,
    ) -> dict[str, JsonValue]:
        """Project only the content leaves of an Agent prompt draft."""
        if type(request_identity_version) is not int:
            raise ValueError("task request identity version is invalid")
        if request_identity_version not in (1, 2):
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        version = request_identity_version
        value = dict(prompt)
        if version == 2:
            return value

        def content(payload: JsonValue) -> JsonValue:
            if not isinstance(payload, Mapping) or not isinstance(payload.get("items"), list):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return {**payload, "items": [project_user_content_identity(item, version=version)
                                         for item in payload["items"]]}

        def native(item: JsonValue) -> JsonValue:
            if isinstance(item, Mapping) and item.get("kind") == "native":
                return {**item, "value": content(item["value"])}
            return item

        if value.get("kind") == "pydantic-user-content-v1":
            value["value"] = content(value["value"])
        elif value.get("kind") in {"task-user-content-v1", "items"}:
            for key in (("items", "intent") if value["kind"] == "task-user-content-v1" else ("items",)):
                items = value[key]
                if not isinstance(items, list):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                value[key] = [native(item) for item in items]
        return value

    def to_mapping(self) -> dict[str, JsonValue]:
        return self.request_identity_payload(request_identity_version=2)

    def prompt_intent_digest(self, *, request_identity_version: int = 2) -> str:
        """Retain the selected prompt identity when its bytes are materialized."""
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
        return canonical_sha256(self.prompt_identity_payload(prompt, request_identity_version=request_identity_version))

    @classmethod
    def _input_identity_payload(cls, body: Mapping[str, JsonValue], version: int) -> dict[str, JsonValue]:
        value = dict(body)
        prompt = value.get("prompt")
        if isinstance(prompt, Mapping):
            value["prompt"] = cls.prompt_identity_payload(prompt, request_identity_version=version)
        view = value.get("accepted_input_view")
        if isinstance(view, Mapping) and isinstance(view.get("prompt"), Mapping):
            value["accepted_input_view"] = {
                **view, "prompt": cls.prompt_identity_payload(view["prompt"], request_identity_version=version),
            }
        return value

    def request_identity_payload(self, *, request_identity_version: int) -> dict[str, JsonValue]:
        value = super().to_mapping()
        if self.task is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if request_identity_version == 1 and self.original_input is not None:
            value["original_input"] = self._input_identity_payload(self.original_input, request_identity_version)
        if self.input_capture is not None:
            return value
        body = self._input_identity_payload(self.input, request_identity_version)
        body["prompt"] = {
            "kind": "task-prompt-intent-v1",
            "digest": self.prompt_intent_digest(request_identity_version=request_identity_version),
        }
        value["input"] = body
        return value


__all__ = ["AgentTaskNode"]
