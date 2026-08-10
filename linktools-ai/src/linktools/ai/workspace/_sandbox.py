#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace filesystem and process execution boundary."""

from typing import Protocol

from ..core import canonical_sha256
from ..errors import AIError, ErrorCode


class Sandbox(Protocol):
    @property
    def fingerprint(self) -> str: ...

    async def read_file(self, path: str) -> str: ...
    async def write_file(self, path: str, content: str) -> None: ...
    async def run(self, command: str) -> str: ...


class DisabledSandbox:
    @property
    def fingerprint(self) -> str:
        return canonical_sha256({"sandbox": "disabled", "version": 1})

    async def read_file(self, path: str) -> str:
        del path
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)

    async def write_file(self, path: str, content: str) -> None:
        del path, content
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)

    async def run(self, command: str) -> str:
        del command
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)


__all__ = ["DisabledSandbox", "Sandbox"]
