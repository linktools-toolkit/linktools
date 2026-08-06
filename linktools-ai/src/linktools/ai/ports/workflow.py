#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Temporal Gateway protocol."""

from typing import Protocol


class WorkflowGateway(Protocol):
    async def start(self, request: object, execution_id: str, workflow_id: str) -> object: ...
    async def update(self, workflow_id: str, name: str, request: object) -> object: ...
    async def query(self, workflow_id: str, name: str, request: "object | None" = None) -> object: ...
    async def cancel(self, workflow_id: str, request: "object | None" = None) -> object: ...


__all__ = ["WorkflowGateway"]
