#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Blob storage protocol."""

from typing import Protocol


class BlobStore(Protocol):
    async def put(self, data: bytes) -> str: ...
    async def get(self, digest: str) -> bytes: ...


__all__ = ["BlobStore"]
