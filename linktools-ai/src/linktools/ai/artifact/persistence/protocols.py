#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ArtifactRecordStore: the access-control + provenance fact source for
artifacts. Lives in the artifact domain (not storage) -- storage owns only
generic object/blob/coordination machinery and never depends on the domains
that consume it."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..models import ArtifactRecord


@runtime_checkable
class ArtifactRecordStore(Protocol):
    """The access-control + provenance fact source for artifacts.

    Every read loads the record by artifact id and checks tenant ownership
    first; a digest alone is never enough to fetch bytes.
    """

    async def put(self, record: "ArtifactRecord") -> "ArtifactRecord": ...

    async def get(
        self, artifact_id: str, *, tenant_id: str
    ) -> "ArtifactRecord | None": ...

    async def delete(self, artifact_id: str, *, tenant_id: str) -> bool: ...


__all__: "list[str]" = ["ArtifactRecordStore"]
