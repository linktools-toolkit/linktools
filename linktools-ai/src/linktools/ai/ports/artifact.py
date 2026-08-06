#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Artifact delivery and metadata protocols."""

from typing import Protocol


class ArtifactDelivery(Protocol):
    async def prepare_download(self, artifact_id: str, expires_at: object) -> object: ...
    async def open_stream(self, artifact_id: str) -> object: ...


class ArtifactRepository(Protocol):
    async def list(self, execution_id: str, cursor: "str | None", limit: int) -> object: ...
    async def get(self, artifact_id: str) -> "object | None": ...


__all__ = ["ArtifactDelivery", "ArtifactRepository"]
