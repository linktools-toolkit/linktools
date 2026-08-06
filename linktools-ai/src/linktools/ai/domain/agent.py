#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Agent Release and immutable bundle descriptors."""

from pydantic import BaseModel, ConfigDict, Field

from .execution import ExecutionProfile


class AgentBundleDescriptor(BaseModel):
    """Fixed identity of one generated Agent Bundle."""

    model_config = ConfigDict(frozen=True)

    agent_id: str = Field(min_length=1)
    agent_revision: int = Field(ge=1)
    toolset_ids: "tuple[str, ...]" = ()
    output_contract_id: str = Field(min_length=1)
    deps_contract_id: str = Field(min_length=1)
    build_id: str = Field(min_length=1)


class AgentRelease(BaseModel):
    """A fixed, enabled version of an Agent Spec."""

    model_config = ConfigDict(frozen=True)

    agent_id: str
    revision: int = Field(ge=1)
    spec_uri: str
    spec_sha256: str
    allowed_profiles: "frozenset[ExecutionProfile]"
    policy_id: str
    output_contract_id: str
    output_contract_version: int = Field(ge=1)
    deps_contract_id: str
    deps_contract_version: int = Field(ge=1)
    enabled: bool = True


BundleDescriptor = AgentBundleDescriptor
