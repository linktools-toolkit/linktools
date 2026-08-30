#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared policy for automatic semantic discovery."""

from dataclasses import dataclass
from pathlib import PurePosixPath


@dataclass(frozen=True, slots=True)
class _DiscoveryPolicy:
    ignore_hidden: bool = True
    ignored_names: frozenset[str] = frozenset()
    ignored_suffixes: tuple[str, ...] = ()

    def ignores(self, path: str) -> bool:
        if not isinstance(path, str):
            raise TypeError("discovery path must be a string")
        parts = PurePosixPath(path).parts
        if not parts:
            return False
        folded = tuple(part.casefold() for part in parts)
        return (
            self.ignore_hidden
            and any(part.startswith(".") for part in parts)
            or any(part in self.ignored_names for part in folded)
            or folded[-1].endswith(self.ignored_suffixes)
        )


DEFAULT_DISCOVERY_POLICY = _DiscoveryPolicy(
    ignored_names=frozenset(
        {"__macosx", "__pycache__", "desktop.ini", "thumbs.db"}
    ),
    ignored_suffixes=(".pyc", ".pyo"),
)


__all__ = ["DEFAULT_DISCOVERY_POLICY"]
