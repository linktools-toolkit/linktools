#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Immutable model-plan validation."""

from ...domain.model import ExecutionModelPlan


class ModelPolicyService:
    def resolve_plan(self, plan: ExecutionModelPlan, agent_id: str) -> ExecutionModelPlan:
        if not plan.route_for(agent_id) or any(route.max_output_tokens <= 0 for route in plan.route_for(agent_id)):
            raise ValueError("model plan has no bounded route")
        return plan


__all__ = ["ModelPolicyService"]
