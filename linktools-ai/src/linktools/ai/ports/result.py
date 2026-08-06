#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Final result protocol."""

from typing import Protocol


class ResultRepository(Protocol):
    async def commit(self, result: object) -> object: ...
    async def get(self, execution_id: str) -> "object | None": ...


__all__ = ["ResultRepository"]
