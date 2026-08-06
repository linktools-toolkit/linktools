#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"Deterministic task Workflow kernel."


class TaskWorkflow:
    def __init__(self, activities: object) -> None:
        self._activities = activities

    async def run(self, request: object) -> object:
        return await self._activities.execute(request)
