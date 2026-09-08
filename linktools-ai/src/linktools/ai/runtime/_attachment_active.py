#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Process-local projection of one Execution's formally committed active attachments."""

from collections.abc import Sequence

from ..core import canonical_sha256
from ..errors import AIError, ErrorCode
from ._attachment_adapter import _semantic_entry_digest
from .state import (
    AttachmentResult,
    Locator,
    ModelExposureEntry,
    RuntimeDomain,
    record_key_digest,
    semantic_attachment_entry,
)


class AttachmentActiveSet:
    """Maintain first-activation order from durable input/read facts for one E."""

    def __init__(
        self,
        *,
        namespace: str,
        tenant_id: str,
        execution_id: str,
        initial: Sequence[ModelExposureEntry] = (),
    ) -> None:
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("namespace is required")
        if not isinstance(tenant_id, str) or not tenant_id:
            raise ValueError("tenant_id is required")
        if not isinstance(execution_id, str) or not execution_id:
            raise ValueError("execution_id is required")
        self._namespace = namespace
        self._tenant_id = tenant_id
        self._execution_id = execution_id
        self._values: list[ModelExposureEntry] = []
        self._by_id: dict[str, ModelExposureEntry] = {}
        self.extend(initial)

    def entries(self) -> tuple[ModelExposureEntry, ...]:
        return tuple(self._values)

    def extend(self, values: Sequence[ModelExposureEntry]) -> None:
        for value in values:
            if not isinstance(value, ModelExposureEntry):
                raise TypeError("active attachment values must be ModelExposureEntry")
            existing = self._by_id.get(value.activation_id)
            if existing is not None:
                if existing != value:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                continue
            self._by_id[value.activation_id] = value
            self._values.append(value)

    async def add_committed_read(
        self,
        tool_operation_id: str,
        result: AttachmentResult,
    ) -> None:
        if not isinstance(tool_operation_id, str) or not tool_operation_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if not isinstance(result, AttachmentResult):
            raise TypeError("result must be AttachmentResult")
        key = record_key_digest(
            self._namespace,
            self._tenant_id,
            RuntimeDomain.RECOVERY.value,
            "tool_operation",
            tool_operation_id,
        ).hex()
        source = Locator("state:recovery", "records", key)
        semantic_digest = _semantic_entry_digest(
            semantic_attachment_entry(result.entry)
        )
        value = ModelExposureEntry(
            canonical_sha256(
                {
                    "version": 1,
                    "execution_id": self._execution_id,
                    "source": source.to_json(),
                    "slot": 0,
                    "entry_digest": semantic_digest,
                }
            ),
            source,
            0,
            result.entry,
        )
        self.extend((value,))

    def authorize(self, values: Sequence[ModelExposureEntry]) -> None:
        for value in values:
            if self._by_id.get(value.activation_id) != value:
                raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)


__all__: tuple[str, ...] = ()
