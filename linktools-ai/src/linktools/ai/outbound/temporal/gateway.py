#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"Temporal Workflow Gateway; no Workflow logic belongs here."


class TemporalWorkflowGateway:
    def __init__(self, client: object) -> None:
        self._client = client

    async def start(self, request: object, execution_id: str, workflow_id: str) -> object:
        return await self._client.start_workflow(workflow_id, request, id=workflow_id)

    async def update(self, workflow_id: str, name: str, request: object) -> object:
        return await self._client.get_workflow_handle(workflow_id).execute_update(name, request)

    async def query(self, workflow_id: str, name: str, request: "object | None" = None) -> object:
        return await self._client.get_workflow_handle(workflow_id).query(name, request)

    async def cancel(self, workflow_id: str, request: "object | None" = None) -> object:
        return await self._client.get_workflow_handle(workflow_id).cancel()


__all__ = ["TemporalWorkflowGateway"]
