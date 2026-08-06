#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Fail-closed typed Activity binding scope."""

from contextvars import ContextVar
from typing import TypeVar

from ..foundation.errors import ErrorCode, LinktoolsAIError

T = TypeVar("T")


class ActivityScope:
    _current: "ContextVar[object | None]" = ContextVar("linktools_ai_activity_scope", default=None)

    @classmethod
    def bind(cls, binding: object) -> object:
        return cls._current.set(binding)

    @classmethod
    def reset(cls, token: object) -> None:
        cls._current.reset(token)

    @classmethod
    def current(cls, expected: "type[T]") -> T:
        binding = cls._current.get()
        if not isinstance(binding, expected):
            raise LinktoolsAIError(ErrorCode.ACTIVITY_SCOPE_REQUIRED, "expected Activity binding is not installed")
        return binding


__all__ = ["ActivityScope"]
