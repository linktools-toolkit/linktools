#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable cursor pagination values."""

from dataclasses import dataclass
from typing import Generic, TypeVar

ItemT = TypeVar("ItemT")


@dataclass(frozen=True, slots=True)
class Page(Generic[ItemT]):
    items: "tuple[ItemT, ...]"
    next_cursor: "str | None" = None


__all__ = ["Page"]
