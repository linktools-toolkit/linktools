#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Process-local entry-point metadata snapshot."""

from functools import lru_cache
from typing import TYPE_CHECKING

try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata

if TYPE_CHECKING:
    from typing import Any


@lru_cache(maxsize=1)
def get_entry_points() -> "Any":
    return metadata.entry_points()


def select_entry_points(group: str) -> "tuple[Any, ...]":
    entries = get_entry_points()
    select = getattr(entries, "select", None)
    if select is not None:
        return tuple(select(group=group))
    if isinstance(entries, dict):
        return tuple(entries.get(group, ()))
    return tuple(entry for entry in entries if getattr(entry, "group", None) == group)
