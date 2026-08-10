#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace filesystem and process execution boundary."""

from typing import Protocol


class Sandbox(Protocol):
    @property
    def fingerprint(self) -> str: ...

    async def read_file(self, path: str) -> str: ...
    async def write_file(self, path: str, content: str) -> None: ...
    async def run(self, command: str) -> str: ...


__all__ = ["Sandbox"]
