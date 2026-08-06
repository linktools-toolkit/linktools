#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Principal lookup boundary."""

from typing import Protocol, runtime_checkable

from .value import Principal


@runtime_checkable
class PrincipalProvider(Protocol):
    async def current(self) -> Principal: ...


__all__ = ["PrincipalProvider"]
