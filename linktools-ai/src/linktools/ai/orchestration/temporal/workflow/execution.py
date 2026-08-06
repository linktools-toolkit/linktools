#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"Deterministic execution Workflow kernel."


class ExecutionWorkflow:
    def __init__(self, activities: object) -> None:
        self._activities = activities

    async def run(self, request: object) -> object:
        return await self._activities.execute(request)
