#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Host-authorized access to recoverable input content."""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from copy import deepcopy
from typing import Generic, TypeVar

from pydantic_ai.capabilities import AbstractCapability

from ..core import PromptLimits
from ..errors import AIError, ErrorCode
from ..model import ReferencedContent, ReferencedInputOrigin
from ..spec import CapabilityRef
from ..model.messages import decode_user_content_item, encode_user_content_item
from ._context import AgentContext

AppT = TypeVar("AppT")


@dataclass(frozen=True, slots=True)
class ReferencedInputResult:
    """One authorized batch result, identified by its trusted source descriptor."""

    content: ReferencedContent
    data: bytes


class ReferencedInputsCapability(AbstractCapability[AgentContext[AppT]], Generic[AppT]):
    """Bind an authorized input reader through an Agent's durable id/revision.

    Register with ``scope='agent'``. Each callback must authorize the original
    execution principal and input membership, validate MIME against trusted
    source metadata, and fetch the exact immutable version. A reference alone
    is not authorization. Authorization is repeated on every read; this owner
    never caches grants or bytes.

    ``authorized_load_many`` receives one ordered, same-source batch. It can
    authorize and fetch admission metadata together instead of performing one
    metadata query per file. Return a result per occurrence in request order,
    including duplicates, using the descriptor from the trusted metadata.
    Without that callback ``load_many`` retains scalar authorization semantics.
    A batch-only reader also supports ``load`` as a one-item batch. Optional
    ``limits`` reuse PromptLimits for explicit reads; Runtime independently
    enforces the entire provider request budget before hydration.

    The callback owns temporary downloads and removes them on success, failure
    and cancellation before returning. Runtime borrows it; registering a loader
    does not create a local-file handoff or grant permission to invoke a tool.
    """

    def __init__(
        self,
        authorized_load: Callable[[AgentContext[AppT], ReferencedContent], Awaitable[bytes]] | None = None,
        *,
        authorized_load_many: Callable[
            [AgentContext[AppT], tuple[ReferencedContent, ...]],
            Awaitable[Sequence[ReferencedInputResult]],
        ] | None = None,
        limits: PromptLimits | None = None,
    ) -> None:
        if authorized_load is not None and not callable(authorized_load):
            raise TypeError("authorized_load must be callable or None")
        if authorized_load is None and authorized_load_many is None:
            raise ValueError("an authorized input loader is required")
        if authorized_load_many is not None and not callable(authorized_load_many):
            raise TypeError("authorized_load_many must be callable")
        if limits is not None and not isinstance(limits, PromptLimits):
            raise TypeError("limits must be PromptLimits or None")
        self._authorized_load = authorized_load
        self._authorized_load_many = authorized_load_many
        self._limits = limits

    def validate_batch(self, contents: Sequence[ReferencedContent]) -> tuple[ReferencedContent, ...]:
        """Snapshot descriptor values and enforce the complete read's input budget."""
        if isinstance(contents, (str, bytes)) or not isinstance(contents, Sequence):
            raise TypeError("contents must be a sequence of ReferencedContent")
        if self._limits is not None and len(contents) > self._limits.max_binary_input_parts:
            raise AIError(ErrorCode.CONTEXT_PAYLOAD_TOO_LARGE, retryable=False)
        result = []
        total = 0
        for content in contents:
            if not isinstance(content, ReferencedContent):
                raise TypeError("contents must contain ReferencedContent")
            value = decode_user_content_item(encode_user_content_item(content))
            if not isinstance(value, ReferencedContent):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID, retryable=False)
            value.content = content.content
            value.metadata = deepcopy(content.metadata)
            total += value.ref.size
            result.append(value)
        if self._limits is not None and total > self._limits.max_binary_input_bytes:
            raise AIError(ErrorCode.CONTEXT_PAYLOAD_TOO_LARGE, retryable=False)
        return tuple(result)

    async def load(self, context: AgentContext[AppT], content: ReferencedContent) -> bytes:
        """Authorize a scalar read without retaining grants or returned bytes."""
        if self._authorized_load is None:
            return (await self.load_many(context, (content,)))[0]
        snapshot, = self.validate_batch((content,))
        data = await self._authorized_load(context, content)
        self._verify(snapshot, data)
        return data

    async def load_many(
        self, context: AgentContext[AppT], contents: Sequence[ReferencedContent],
    ) -> tuple[bytes, ...]:
        """Authorize one ordered same-source batch, then validate every result."""
        snapshots = self.validate_batch(contents)
        if not snapshots:
            return ()
        if any((content.source, content.origin) != (snapshots[0].source, snapshots[0].origin)
               for content in snapshots):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID, retryable=False,
                          safe_details={"reason": "referenced_input_source_mismatch"})
        if self._authorized_load_many is None:
            assert self._authorized_load is not None
            result = []
            for expected, content in zip(snapshots, contents, strict=True):
                data = await self._authorized_load(context, content)
                self._verify(expected, data)
                result.append(data)
            return tuple(result)
        # Preserve independent expected values: callback mutation cannot rewrite
        # the requested MIME, version, order or integrity contract.
        requested = self.validate_batch(snapshots)
        loaded = await self._authorized_load_many(context, requested)
        if (not isinstance(loaded, Sequence) or isinstance(loaded, (str, bytes))
                or len(loaded) != len(snapshots)):
            self._mismatch("referenced_input_batch_cardinality")
        result = []
        for expected, value in zip(snapshots, loaded, strict=True):
            if not isinstance(value, ReferencedInputResult) or value.content != expected:
                self._mismatch("referenced_input_batch_descriptor")
            self._verify(expected, value.data)
            result.append(value.data)
        return tuple(result)

    @classmethod
    async def load_selected(
        cls, context: AgentContext[AppT], contents: Sequence[ReferencedContent],
        sources: Mapping[CapabilityRef, "ReferencedInputsCapability[AppT]"],
    ) -> tuple[bytes, ...]:
        """Route selected sources by original producer and restore occurrence order."""
        groups: dict[tuple[CapabilityRef, ReferencedInputOrigin | None], list[int]] = {}
        source_contents: dict[CapabilityRef, list[ReferencedContent]] = {}
        for index, content in enumerate(contents):
            if content.source not in sources:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID,
                              safe_details={"reason": "referenced_input_source_mismatch"})
            groups.setdefault((content.source, content.origin), []).append(index)
            source_contents.setdefault(content.source, []).append(content)
        # Splitting a selected source by producer must not bypass its read budget.
        for source, values in source_contents.items():
            sources[source].validate_batch(values)
        result = [b""] * len(contents)
        for (source, _origin), indices in groups.items():
            loaded = await sources[source].load_many(context, tuple(contents[index] for index in indices))
            for index, data in zip(indices, loaded, strict=True):
                result[index] = data
        return tuple(result)

    @classmethod
    def _mismatch(cls, reason: str) -> None:
        raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH, retryable=False,
                      safe_details={"reason": reason})

    @classmethod
    def _verify(cls, content: ReferencedContent, data: bytes) -> None:
        if not isinstance(data, bytes):
            cls._mismatch("referenced_input_bytes_required")
        try:
            content.ref.verify_content(data)
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_INTEGRITY_ERROR:
                raise
            raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH, retryable=False) from error


__all__ = ["ReferencedInputResult", "ReferencedInputsCapability"]
