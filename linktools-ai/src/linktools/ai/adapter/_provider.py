#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model provider adapter protocol."""

from typing import Protocol


class ProviderClient(Protocol):
    async def complete(self, prompt: str) -> str: ...


__all__ = ["ProviderClient"]
