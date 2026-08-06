#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Sandbox and checkpoint protocols."""

from typing import Protocol


class SandboxProvisioner(Protocol):
    async def create(self, request: object) -> object: ...
    async def inspect(self, lease_id: str) -> object: ...
    async def destroy(self, lease_id: str) -> None: ...


class SandboxCommandExecutor(Protocol):
    async def execute(self, lease_id: str, command: object) -> object: ...


class WorkspaceCheckpointPort(Protocol):
    async def capture(self, workspace: object) -> object: ...
    async def restore(self, checkpoint: object) -> object: ...


__all__ = ["SandboxCommandExecutor", "SandboxProvisioner", "WorkspaceCheckpointPort"]
