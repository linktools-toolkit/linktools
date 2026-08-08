#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Serializable Agent dependencies; no live client objects."""

from pydantic import BaseModel, ConfigDict


class AgentDeps(BaseModel):
    model_config = ConfigDict(frozen=True)
    execution_id: str
    tenant_principal_ref: str
    model_plan_id: str
    budget_id: str
    prompt_snapshot_id: str
    repo_context_snapshot_id: "str | None" = None
    live_subject: "str | None" = None


__all__ = ["AgentDeps"]
