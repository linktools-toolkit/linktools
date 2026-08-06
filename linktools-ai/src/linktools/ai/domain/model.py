#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Fixed model route metadata."""

from pydantic import BaseModel, ConfigDict, Field


class ModelRoute(BaseModel):
    """Startup-registered provider route."""

    model_config = ConfigDict(frozen=True)

    route_id: str
    provider: str
    model: str
    max_output_tokens: int = Field(ge=1)


class ExecutionModelPlan(BaseModel):
    """Execution-lifetime model route decision."""

    model_config = ConfigDict(frozen=True)

    plan_id: str
    credential_route_id: str
    agent_routes: "dict[str, tuple[ModelRoute, ...]]"
    price_table_version: str
    policy_sha256: str

    def route_for(self, agent_id: str) -> 'tuple[ModelRoute, ...]':
        """Return fixed routes for one Agent."""
        return self.agent_routes.get(agent_id, ())
