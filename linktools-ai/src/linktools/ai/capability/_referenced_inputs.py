#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Host-authorized access to recoverable input content."""

from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

from pydantic_ai.capabilities import AbstractCapability

from ..errors import AIError, ErrorCode
from ..model import ReferencedContent
from ._context import AgentContext

AppT = TypeVar("AppT")


class ReferencedInputsCapability(AbstractCapability[AgentContext[AppT]], Generic[AppT]):
    """Bind one authorized input loader through an Agent's durable id/revision.

    Register with ``scope='agent'``. The callback must authorize the original
    execution principal against the host's admitted input set, validate MIME
    against its trusted descriptor, and fetch the exact immutable version.
    It may route multiple configured resources, but must never probe unrelated
    readers or substitute a newer version. Multiple selected input capabilities
    are routed by the reference's exact source id and revision; one contribution
    must not contain multiple loaders for that same source. It owns temporary
    download files and removes them on success, failure and cancellation before
    returning.
    Runtime borrows the callback; it neither creates nor deletes provider files.
    """

    def __init__(
        self,
        authorized_load: Callable[[AgentContext[AppT], ReferencedContent], Awaitable[bytes]],
    ) -> None:
        if not callable(authorized_load):
            raise TypeError("authorized_load must be callable")
        self._authorized_load = authorized_load

    async def load(self, context: AgentContext[AppT], content: ReferencedContent) -> bytes:
        """Authorize and read one descriptor without retaining the returned bytes."""
        data = await self._authorized_load(context, content)
        try:
            content.ref.verify_content(data)
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_INTEGRITY_ERROR:
                raise
            # Returned input bytes disagree with their descriptor; this does
            # not establish corruption of Runtime's authoritative storage.
            raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH, retryable=False) from error
        return data


__all__ = ["ReferencedInputsCapability"]
