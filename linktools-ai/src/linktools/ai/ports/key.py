#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Opaque payload and blob key-management protocol."""

from typing import Protocol


class KeyManagement(Protocol):
    async def wrap(self, key: bytes) -> object: ...
    async def unwrap(self, reference: object) -> bytes: ...
    async def rotate(self, reference: object) -> object: ...


__all__ = ["KeyManagement"]
