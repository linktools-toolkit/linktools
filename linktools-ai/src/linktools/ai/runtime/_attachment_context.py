#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Composition-time bridge for managed attachment admission."""

from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator

from ..errors import AIError, ErrorCode
from .state import PreparedInput


@dataclass(frozen=True, slots=True)
class ManagedAdmission:
    scope: str
    idempotency_key: str
    prepared: PreparedInput


_ScopeFactory = Callable[[ManagedAdmission], AbstractContextManager[None]]
_scope_factory: _ScopeFactory | None = None


def bind_admission_scope(factory: _ScopeFactory) -> None:
    """Bind the one managed-admission owner at Runtime composition time."""
    global _scope_factory
    if not callable(factory):
        raise TypeError("factory must be callable")
    if _scope_factory is not None and _scope_factory is not factory:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    _scope_factory = factory


@contextmanager
def admission_scope(value: ManagedAdmission) -> Iterator[None]:
    """Enter the bound managed-admission transaction context."""
    factory = _scope_factory
    if factory is None:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    with factory(value):
        yield


__all__ = ["ManagedAdmission", "admission_scope", "bind_admission_scope"]
