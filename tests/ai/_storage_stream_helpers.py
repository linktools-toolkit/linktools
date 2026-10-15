#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect asynchronous storage results for whole-result assertions."""

from collections.abc import AsyncIterator
from typing import TypeVar

ValueT = TypeVar("ValueT")


async def collect_values(values: AsyncIterator[ValueT]) -> tuple[ValueT, ...]:
    return tuple([value async for value in values])
