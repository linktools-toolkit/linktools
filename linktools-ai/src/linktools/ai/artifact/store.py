#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Structural artifact-store contract implemented directly by stores."""


from collections.abc import AsyncIterable, AsyncIterator
from typing import Protocol

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import ArtifactRecord

class ArtifactStore(Protocol):
    async def put(
        self, *, record: "ArtifactRecord", content: "AsyncIterable[bytes]"
    ) -> "ArtifactRecord": ...

    async def get_record(
        self, artifact_id: str, *, tenant_id: str
    ) -> "ArtifactRecord | None": ...

    def open(
        self, artifact_id: str, *, tenant_id: str
    ) -> "AsyncIterator[bytes]": ...

    async def delete(self, artifact_id: str, *, tenant_id: str) -> None: ...


__all__ = ["ArtifactStore"]
