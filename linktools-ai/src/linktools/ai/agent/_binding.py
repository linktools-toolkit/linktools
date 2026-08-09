#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolved Agent definition binding and stable behavior digest."""

from dataclasses import dataclass

from pydantic import BaseModel

from ..core import canonical_sha256
from ..capability import MCPServerSpec, MCPToolProvider, Sandbox, SkillProvider, SkillSpec, ToolPolicy
from ..errors import ErrorCode, AIError
from ..model import ModelResolver, ModelRoute
from ..observe import MiddlewarePipeline
from ..spec import AgentSpec, PromptSpec
from ..spec import OutputTypeRegistry


@dataclass(frozen=True, slots=True)
class AgentBinding:
    spec: AgentSpec
    prompt: PromptSpec
    spec_fingerprint: str
    prompt_fingerprint: str
    model_registry_revision: int
    output_schema_fingerprint: str
    capability_manifest_digest: str
    tool_policy_fingerprint: str
    sandbox_fingerprint: str
    middleware_fingerprint: str

    def __post_init__(self) -> None:
        if (
            self.model_registry_revision < 0
            or not self.spec_fingerprint
            or not self.prompt_fingerprint
            or not self.output_schema_fingerprint
            or not self.capability_manifest_digest
            or not self.tool_policy_fingerprint
            or not self.sandbox_fingerprint
            or not self.middleware_fingerprint
        ):
            raise ValueError("Agent binding is incomplete")

    @property
    def digest(self) -> str:
        return canonical_sha256(
            {
                "agent_id": self.spec.id,
                "agent_revision": self.spec.revision,
                "prompt_id": self.prompt.id,
                "prompt_revision": self.prompt.revision,
                "spec_fingerprint": self.spec_fingerprint,
                "prompt_fingerprint": self.prompt_fingerprint,
                "model_registry_revision": self.model_registry_revision,
                "output_schema_fingerprint": self.output_schema_fingerprint,
                "output_schema_id": self.spec.output_schema,
                "output_schema_revision": self.spec.output_schema_revision,
                "capability_manifest_digest": self.capability_manifest_digest,
                "tool_policy_fingerprint": self.tool_policy_fingerprint,
                "sandbox_fingerprint": self.sandbox_fingerprint,
                "middleware_fingerprint": self.middleware_fingerprint,
            }
        )


@dataclass(frozen=True, slots=True)
class BindingDependencies:
    model_resolver: ModelResolver
    skill_provider: SkillProvider
    mcp_provider: MCPToolProvider
    middleware: MiddlewarePipeline
    sandbox: Sandbox
    tool_policy: ToolPolicy
    output_types: OutputTypeRegistry


@dataclass(frozen=True, slots=True)
class BindingExecutionPlan:
    binding: AgentBinding
    model_route: ModelRoute
    output_type: "type[BaseModel]"
    skills: "tuple[SkillSpec, ...]"
    mcp_servers: "tuple[MCPServerSpec, ...]"

    def __post_init__(self) -> None:
        if self.model_route.route_id != self.binding.spec.model:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if canonical_sha256(self.output_type.model_json_schema()) != self.binding.output_schema_fingerprint:
            raise AIError(ErrorCode.OUTPUT_SCHEMA_DRIFT)
        skill_revisions = {(item.id, item.revision) for item in self.skills}
        mcp_revisions = {(item.id, item.revision) for item in self.mcp_servers}
        for feature in self.binding.spec.features:
            if not feature.required:
                continue
            if feature.kind == "skill" and not any(item[0] == feature.id and (feature.revision is None or item[1] == feature.revision) for item in skill_revisions):
                raise AIError(ErrorCode.FEATURE_REQUIRED_MISSING)
            if feature.kind == "mcp" and not any(item[0] == feature.id and (feature.revision is None or item[1] == feature.revision) for item in mcp_revisions):
                raise AIError(ErrorCode.FEATURE_REQUIRED_MISSING)


class BindingExecutionRegistry:
    """Process-local immutable plan registry keyed by binding digest."""

    def __init__(self) -> None:
        self._plans: dict[str, BindingExecutionPlan] = {}

    def register(self, plan: BindingExecutionPlan) -> None:
        digest = plan.binding.digest
        previous = self._plans.get(digest)
        if previous is not None and previous != plan:
            raise AIError(ErrorCode.BINDING_CONFLICT)
        self._plans[digest] = plan

    def resolve(self, digest: str) -> BindingExecutionPlan:
        try:
            return self._plans[digest]
        except KeyError as error:
            raise AIError(ErrorCode.BINDING_NOT_REGISTERED) from error


def build_binding_plan(spec: AgentSpec, prompt: PromptSpec, *, dependencies: BindingDependencies) -> BindingExecutionPlan:
    route = dependencies.model_resolver.resolve(spec.model)
    spec_fingerprint = canonical_sha256(
        {
            "id": spec.id,
            "revision": spec.revision,
            "model": spec.model,
            "features": [
                {"kind": feature.kind, "id": feature.id, "revision": feature.revision, "required": feature.required, "config": dict(feature.config)}
                for feature in spec.features
            ],
            "output_schema": spec.output_schema,
            "output_schema_revision": spec.output_schema_revision,
            "instructions": list(spec.instructions),
        }
    )
    prompt_fingerprint = canonical_sha256(
        {"id": prompt.id, "revision": prompt.revision, "system": prompt.system, "instructions": list(prompt.instructions), "variables": list(prompt.variables)}
    )
    output_schema_fingerprint = dependencies.output_types.fingerprint(spec.output_schema, spec.output_schema_revision)
    capabilities: list[dict[str, object]] = []
    skills: list[SkillSpec] = []
    mcp_servers: list[MCPServerSpec] = []
    for feature in spec.features:
        try:
            if feature.kind == "skill":
                resolved = dependencies.skill_provider.resolve_ref(feature.id, feature.revision)
                if feature.revision is not None and resolved.revision != feature.revision:
                    raise AIError(ErrorCode.FEATURE_REQUIRED_MISSING)
                skills.append(resolved)
                fingerprint = canonical_sha256({"id": resolved.id, "revision": resolved.revision, "content": resolved.content})
                provider_digest = dependencies.skill_provider.manifest()
                resolved_revision = resolved.revision
            elif feature.kind == "mcp":
                resolved = dependencies.mcp_provider.resolve_ref(feature.id, feature.revision)
                if feature.revision is not None and resolved.revision != feature.revision:
                    raise AIError(ErrorCode.FEATURE_REQUIRED_MISSING)
                mcp_servers.append(resolved)
                fingerprint = canonical_sha256({"id": resolved.id, "revision": resolved.revision, "command": resolved.command, "args": list(resolved.args)})
                provider_digest = dependencies.mcp_provider.manifest()
                resolved_revision = resolved.revision
            elif feature.kind not in {"tool", "subagent", "sandbox", "middleware"}:
                raise AIError(ErrorCode.FEATURE_REQUIRED_MISSING)
            else:
                resolved_revision = feature.revision or 1
                fingerprint = canonical_sha256({"kind": feature.kind, "id": feature.id, "revision": resolved_revision, "config": dict(feature.config)})
                provider_digest = ""
        except AIError as error:
            if feature.required:
                raise AIError(ErrorCode.FEATURE_REQUIRED_MISSING) from error
            resolved_revision = 0
            fingerprint = "UNRESOLVED"
            provider_digest = "UNRESOLVED"
        capabilities.append({"kind": feature.kind, "id": feature.id, "requested_revision": feature.revision or 0, "resolved_revision": resolved_revision, "config_digest": canonical_sha256(dict(feature.config)), "fingerprint": fingerprint, "provider_manifest_digest": provider_digest, "required": feature.required})
    capability_manifest_digest = canonical_sha256({"features": sorted(capabilities, key=lambda value: (str(value["kind"]), str(value["id"]), int(value["resolved_revision"])))})
    binding = AgentBinding(
        spec,
        prompt,
        spec_fingerprint,
        prompt_fingerprint,
        dependencies.model_resolver.snapshot().revision,
        output_schema_fingerprint,
        capability_manifest_digest,
        dependencies.tool_policy.fingerprint,
        dependencies.sandbox.fingerprint,
        dependencies.middleware.fingerprint,
    )
    plan = BindingExecutionPlan(binding, route, dependencies.output_types.resolve(spec.output_schema, spec.output_schema_revision), tuple(skills), tuple(mcp_servers))
    return plan


__all__ = ["AgentBinding", "BindingDependencies", "BindingExecutionPlan", "BindingExecutionRegistry", "build_binding_plan"]
