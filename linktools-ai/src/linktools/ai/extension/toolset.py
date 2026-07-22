#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extension toolsets: read-only discovery tools exposed only when
an agent declares ``extension-asset`` / ``extension-entrypoint``. They enforce
pagination + size limits at the provider layer and an allowlist at the toolset
layer (an agent may only touch extensions it declared). Execution tools
(``call_extension_entrypoint``) stay opt-in and are wired through the subagent path."""

from typing import Any, Mapping

from pydantic_ai.toolsets import FunctionToolset

from ..errors import (
    ExtensionEntrypointDeniedError,
    ExtensionEntrypointNotFoundError,
    ExtensionContentAccessDeniedError,
)
from .spec import ExtensionContentSource
from ..run.identity import ParentRunIdentity
from .entrypoint import EntrypointRef
from .provider import DEFAULT_LIST_LIMIT, DEFAULT_MAX_READ_BYTES
from .resolver import EntrypointResolver
from .scope import ExtensionScope


def _check_allowed(
    extension_id: str, allowed: "Mapping[str, ExtensionScope]"
) -> ExtensionScope:
    scope = allowed.get(extension_id)
    if scope is None:
        raise ExtensionContentAccessDeniedError(
            f"extension {extension_id!r} is not declared in this agent's tools"
        )
    return scope


def build_extension_resource_toolset(
    provider: ExtensionContentSource,
    *,
    allowed: "Mapping[str, ExtensionScope]",
    max_resources_per_list: int = DEFAULT_LIST_LIMIT,
    max_read_bytes: int = DEFAULT_MAX_READ_BYTES,
    emit=None,
) -> FunctionToolset:
    """Level-1 read tools: list_extension_content / read_extension_content.
    ``allowed`` maps a declared extension_id to its scope; undeclared ids are
    refused before any filesystem access."""
    from ..events.payloads import ExtensionContentListed, ExtensionContentRead

    toolset: FunctionToolset = FunctionToolset()
    cap_list = max_resources_per_list
    cap_read = max_read_bytes

    async def list_extension_content(
        extension_id: str,
        path: str = "",
        limit: int = cap_list,
        cursor: "str | None" = None,
    ) -> "dict[str, Any]":
        """List files under a path in a declared extension (paginated)."""
        scope = _check_allowed(extension_id, allowed)
        effective_limit = min(limit, cap_list) if cap_list else limit
        result = await provider.list_entries(
            scope,
            path,
            limit=effective_limit,
            cursor=cursor,
        )
        if emit is not None:
            await emit(
                ExtensionContentListed(
                    extension_id=extension_id, path=path, count=len(result.items)
                )
            )
        return result.model_dump()

    async def read_extension_content(
        extension_id: str,
        path: str,
        max_bytes: "int | None" = None,
    ) -> "dict[str, Any]":
        """Read one asset from a declared extension (size-clamped)."""
        from .content import ExtensionContentRef

        scope = _check_allowed(extension_id, allowed)
        effective = min(max_bytes, cap_read) if max_bytes is not None else cap_read
        content = await provider.read_content(
            ExtensionContentRef(scope=scope, path=path),
            max_bytes=effective,
        )
        if emit is not None:
            await emit(
                ExtensionContentRead(
                    extension_id=extension_id,
                    path=path,
                    truncated=bool(content.metadata.get("truncated")),
                )
            )
        return content.model_dump()

    toolset.add_function(list_extension_content)
    toolset.add_function(read_extension_content)
    return toolset


def build_extension_entrypoint_toolset(
    resolver: EntrypointResolver,
    *,
    allowed: "Mapping[str, ExtensionScope]",
    allowed_kinds: "tuple[str, ...]" = ("agent",),
    allowed_names: "tuple[str, ...] | None" = None,
    expose_call_tool: bool = False,
    max_entrypoints_per_list: int = DEFAULT_LIST_LIMIT,
    emit=None,
    executor=None,
    parent: "ParentRunIdentity | None" = None,
) -> FunctionToolset:
    """Level-1 list tool for extension entrypoints (``list_extension_entrypoints``).
    Calling an entrypoint is opt-in (``expose_call_tool``) and is wired through
    the subagent runner elsewhere; here it is reserved."""
    from ..events.payloads import ExtensionEntrypointListed

    toolset: FunctionToolset = FunctionToolset()
    cap = max_entrypoints_per_list

    async def list_extension_entrypoints(
        extension_id: str,
        kind: "str | None" = None,
        limit: int = cap,
        cursor: "str | None" = None,
    ) -> "dict[str, Any]":
        """List entrypoints (agents/workflows/...) in a declared extension."""
        scope = _check_allowed(extension_id, allowed)
        effective_limit = min(limit, cap) if cap else limit
        result = await resolver.list_entrypoints(
            scope,
            kind=kind,
            limit=effective_limit,
            cursor=cursor,
        )
        if emit is not None:
            await emit(
                ExtensionEntrypointListed(
                    extension_id=extension_id, kind=kind, count=len(result.items)
                )
            )
        return result.model_dump()

    toolset.add_function(list_extension_entrypoints)

    if expose_call_tool:

        async def call_extension_entrypoint(
            extension_id: str,
            kind: str,
            name: str,
            task: str,
            context: "dict[str, Any] | None" = None,
        ) -> "dict[str, Any]":
            """Run an extension entrypoint. Only declared kinds/names are admitted."""
            scope = _check_allowed(extension_id, allowed)
            if kind not in allowed_kinds:
                raise ExtensionEntrypointDeniedError(
                    f"entrypoint kind {kind!r} not allowed for extension {extension_id!r}"
                )
            if allowed_names is not None and name not in allowed_names:
                raise ExtensionEntrypointDeniedError(
                    f"entrypoint {kind}/{name!r} not in allowlist for extension {extension_id!r}"
                )
            # Execution tools must not fake success: without a resolver/executor
            # the entrypoint cannot run, so raise a structured denial.
            if resolver is None:
                raise ExtensionEntrypointNotFoundError(
                    f"extension entrypoint resolution unavailable for {extension_id!r}"
                )
            if executor is None:
                raise ExtensionEntrypointDeniedError(
                    "extension entrypoint execution requires an entrypoint executor"
                )
            if kind != "agent":
                raise ExtensionEntrypointDeniedError(
                    f"only agent entrypoints are executable, got kind {kind!r}"
                )
            ref = EntrypointRef(kind=kind, name=name, scope=scope)
            agent_spec = await resolver.resolve_agent(ref)
            if emit is not None:
                from ..events.payloads import ExtensionEntrypointResolved

                await emit(
                    ExtensionEntrypointResolved(
                        extension_id=extension_id, kind=kind, name=name
                    )
                )
            if parent is None:
                # A tool can only be invoked from inside a running model call;
                # Runtime.run always builds a real ParentRunIdentity. parent=None
                # means assembly happened outside any run -- fail loudly rather
                # than minting an unparented child run with no lineage.
                raise ExtensionEntrypointDeniedError(
                    "call_extension_entrypoint invoked without a parent run "
                    "identity; entrypoint tools cannot run outside a live Run"
                )
            # Reuse the SAME ParentRunIdentity the caller built -- in
            # particular parent.root_run_id, never re-derived here as
            # "= parent_run_id" (that would truncate lineage to one hop
            # whenever this extension entrypoint is itself already nested under
            # a subagent/entrypoint chain).
            result = await executor.execute(
                agent_spec=agent_spec,
                task=task,
                context=context,
                parent=parent,
                scope=scope,
                timeout_seconds=None,
            )
            return result.model_dump()

        toolset.add_function(call_extension_entrypoint)

    return toolset
