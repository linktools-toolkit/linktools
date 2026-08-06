#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Auditable deletion persistence protocol."""

from typing import Protocol


class DeletionRepository(Protocol):
    async def create(self, job: object) -> object: ...
    async def advance(self, deletion_id: str, job: object) -> object: ...
    async def get(self, deletion_id: str) -> "object | None": ...


__all__ = ["DeletionRepository"]
