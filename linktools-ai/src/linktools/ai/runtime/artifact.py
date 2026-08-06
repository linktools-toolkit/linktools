#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Artifact query and download API."""

from typing import Protocol

from ..core import Page, Principal
from .services import ArtifactDownload, ArtifactView


class ArtifactApi(Protocol):
    async def list(self, execution_id: str, *, principal: Principal, cursor: 'str | None' = None, limit: int = 100) -> 'Page[ArtifactView]': ...
    async def get(self, artifact_id: str, *, principal: Principal) -> ArtifactDownload: ...


__all__ = ["ArtifactApi"]
