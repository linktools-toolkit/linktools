#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Temporal Runtime facade with one-to-one action forwarding."""

from ...ports.runtime import Runtime


class TemporalRuntimeFacade(Runtime):
    def __init__(self, actions: object) -> None:
        self._actions = actions

    async def run(self, request: object) -> object: return await self._actions.run(request)
    async def inspect(self, execution_id: str) -> object: return await self._actions.inspect(execution_id)
    async def result(self, execution_id: str) -> object: return await self._actions.result(execution_id)
    async def cancel(self, execution_id: str, request: object) -> object: return await self._actions.cancel(execution_id, request)
    async def retry(self, execution_id: str, request: object) -> object: return await self._actions.retry(execution_id, request)
    async def fork(self, execution_id: str, request: object) -> object: return await self._actions.fork(execution_id, request)
    async def create(self, request: object) -> object: return await self._actions.create(request)
    async def get(self, session_id: str) -> object: return await self._actions.get(session_id)
    async def list(self, request: object) -> object: return await self._actions.list(request)
    async def load(self, session_id: str, request: object) -> object: return await self._actions.load(session_id, request)
    async def resume(self, session_id: str, request: object) -> object: return await self._actions.resume(session_id, request)
    async def update(self, session_id: str, request: object) -> object: return await self._actions.update(session_id, request)
    async def close(self, session_id: str, request: object) -> object: return await self._actions.close(session_id, request)
    async def submit(self, request: object) -> object: return await self._actions.submit(request)
    async def inspect_task(self, task_id: str) -> object: return await self._actions.inspect_task(task_id)
    async def list_tasks(self, request: object) -> object: return await self._actions.list_tasks(request)
    async def retry_task(self, task_id: str, request: object) -> object: return await self._actions.retry_task(task_id, request)
    async def cancel_task(self, task_id: str, request: object) -> object: return await self._actions.cancel_task(task_id, request)
    async def run_evaluation(self, request: object) -> object: return await self._actions.run_evaluation(request)
    async def inspect_evaluation(self, evaluation_id: str) -> object: return await self._actions.inspect_evaluation(evaluation_id)
    async def compare_evaluation(self, request: object) -> object: return await self._actions.compare_evaluation(request)
    async def snapshot(self, evaluation_id: str) -> object: return await self._actions.snapshot(evaluation_id)
    async def replay(self, snapshot_id: str, request: object) -> object: return await self._actions.replay(snapshot_id, request)


__all__ = ["TemporalRuntimeFacade"]
