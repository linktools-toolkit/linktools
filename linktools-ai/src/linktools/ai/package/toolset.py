#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Package toolsets: read-only discovery tools exposed only when
an agent declares ``package-resource`` / ``package-entrypoint``. They enforce
pagination + size limits at the provider layer and an allowlist at the toolset
layer (an agent may only touch packages it declared). Execution tools
(``call_package_entrypoint``) stay opt-in and are wired through the subagent
path in a later phase."""

from typing import Any, Mapping

from pydantic_ai.toolsets import FunctionToolset

from ..errors import PackageResourceAccessDeniedError, PackageEntrypointDeniedError
from .entrypoint import EntrypointRef
from .provider import DEFAULT_LIST_LIMIT, DEFAULT_MAX_READ_BYTES, DirectoryPackageResourceProvider
from .resolver import DirectoryEntrypointResolver
from .scope import PackageScope


def _check_allowed(package_id: str, allowed: "Mapping[str, PackageScope]") -> PackageScope:
    scope = allowed.get(package_id)
    if scope is None:
        raise PackageResourceAccessDeniedError(
            f"package {package_id!r} is not declared in this agent's tools"
        )
    return scope


def build_package_resource_toolset(
    provider: DirectoryPackageResourceProvider,
    *,
    allowed: "Mapping[str, PackageScope]",
    max_resources_per_list: int = DEFAULT_LIST_LIMIT,
    max_read_bytes: int = DEFAULT_MAX_READ_BYTES,
    emit=None,
) -> FunctionToolset:
    """Level-1 read tools: list_package_resources / read_package_resource.
    ``allowed`` maps a declared package_id to its scope; undeclared ids are
    refused before any filesystem access."""
    from ..events.payloads import PackageResourceListed, PackageResourceRead

    toolset: FunctionToolset = FunctionToolset()
    cap_list = max_resources_per_list
    cap_read = max_read_bytes

    async def list_package_resources(
        package_id: str, path: str = "", limit: int = cap_list, cursor: "str | None" = None,
    ) -> "dict[str, Any]":
        """List files under a path in a declared package (paginated)."""
        scope = _check_allowed(package_id, allowed)
        effective_limit = min(limit, cap_list) if cap_list else limit
        result = await provider.list_resources(
            scope, path, limit=effective_limit, cursor=cursor,
        )
        if emit is not None:
            await emit(PackageResourceListed(package_id=package_id, path=path, count=len(result.items)))
        return result.model_dump()

    async def read_package_resource(
        package_id: str, path: str, max_bytes: "int | None" = None,
    ) -> "dict[str, Any]":
        """Read one resource from a declared package (size-clamped)."""
        from .resource import ResourceRef
        scope = _check_allowed(package_id, allowed)
        effective = min(max_bytes, cap_read) if max_bytes is not None else cap_read
        content = await provider.read_resource(
            ResourceRef(scope=scope, path=path), max_bytes=effective,
        )
        if emit is not None:
            await emit(PackageResourceRead(package_id=package_id, path=path,
                                           truncated=bool(content.metadata.get("truncated"))))
        return content.model_dump()

    toolset.add_function(list_package_resources)
    toolset.add_function(read_package_resource)
    return toolset


def build_package_entrypoint_toolset(
    resolver: DirectoryEntrypointResolver,
    *,
    allowed: "Mapping[str, PackageScope]",
    allowed_kinds: "tuple[str, ...]" = ("agent",),
    allowed_names: "tuple[str, ...] | None" = None,
    expose_call_tool: bool = False,
    max_entrypoints_per_list: int = DEFAULT_LIST_LIMIT,
    emit=None,
    executor=None,
    parent_run_id: "str | None" = None,
    parent_session_id: "str | None" = None,
) -> FunctionToolset:
    """Level-1 list tool for package entrypoints (``list_package_entrypoints``).
    Calling an entrypoint is opt-in (``expose_call_tool``) and is wired through
    the subagent runner elsewhere; here it is reserved."""
    from ..events.payloads import PackageEntrypointListed

    toolset: FunctionToolset = FunctionToolset()
    cap = max_entrypoints_per_list

    async def list_package_entrypoints(
        package_id: str, kind: "str | None" = None, limit: int = cap, cursor: "str | None" = None,
    ) -> "dict[str, Any]":
        """List entrypoints (agents/workflows/...) in a declared package."""
        scope = _check_allowed(package_id, allowed)
        effective_limit = min(limit, cap) if cap else limit
        result = await resolver.list_entrypoints(
            scope, kind=kind, limit=effective_limit, cursor=cursor,
        )
        if emit is not None:
            await emit(PackageEntrypointListed(package_id=package_id, kind=kind, count=len(result.items)))
        return result.model_dump()

    toolset.add_function(list_package_entrypoints)

    if expose_call_tool:
        async def call_package_entrypoint(
            package_id: str, kind: str, name: str, task: str,
            context: "dict[str, Any] | None" = None,
        ) -> "dict[str, Any]":
            """Run a package entrypoint. Only declared kinds/names are admitted."""
            scope = _check_allowed(package_id, allowed)
            if kind not in allowed_kinds:
                raise PackageEntrypointDeniedError(
                    f"entrypoint kind {kind!r} not allowed for package {package_id!r}"
                )
            if allowed_names is not None and name not in allowed_names:
                raise PackageEntrypointDeniedError(
                    f"entrypoint {kind}/{name!r} not in allowlist for package {package_id!r}"
                )
            # Resolve the scoped agent and delegate to the subagent executor when
            # wired; otherwise return a reserved marker (the gate is still testable).
            ref = EntrypointRef(kind=kind, name=name, scope=scope)
            if executor is None or resolver is None:
                return {"status": "reserved", "ref": ref.internal_key(), "task": task}
            if kind != "agent":
                raise PackageEntrypointDeniedError(
                    f"only agent entrypoints are executable, got kind {kind!r}"
                )
            agent_spec = await resolver.resolve_agent(ref)
            if emit is not None:
                from ..events.payloads import PackageEntrypointResolved
                await emit(PackageEntrypointResolved(
                    package_id=package_id, kind=kind, name=name))
            result = await executor.execute(
                agent_spec=agent_spec, task=task, context=context,
                parent_run_id=parent_run_id, root_run_id=parent_run_id,
                parent_session_id=parent_session_id, scope=scope,
                timeout_seconds=None,
            )
            return result.model_dump()

        toolset.add_function(call_package_entrypoint)

    return toolset
