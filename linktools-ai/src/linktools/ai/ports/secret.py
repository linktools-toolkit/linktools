#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Opaque Secret reference protocol."""

from typing import Protocol


class SecretProvider(Protocol):
    async def resolve(self, reference: str) -> object: ...


__all__ = ["SecretProvider"]
