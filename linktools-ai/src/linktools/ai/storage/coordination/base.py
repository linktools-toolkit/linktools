#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AssetCoordinator: hint-only revision/lock coordination. Never stores asset
content and is never the source of correctness -- the DB (or file backend) always is."""

from typing import AsyncContextManager, Protocol, runtime_checkable


@runtime_checkable
class AssetCoordinator(Protocol):
    async def revision_hint(self) -> "int | None": ...

    async def publish_revision(self, revision: int) -> None: ...

    def lock(self, key: str) -> "AsyncContextManager[None]": ...
