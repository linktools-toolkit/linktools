#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"Deterministic registry Workflow kernel."

from ....bundles.generated import agent


class WorkflowRegistry:
    __pydantic_ai_agents__ = (agent,)

    def __init__(self, activities: object) -> None:
        self._activities = activities

    async def run(self, request: object) -> object:
        return await self._activities.execute(request)
