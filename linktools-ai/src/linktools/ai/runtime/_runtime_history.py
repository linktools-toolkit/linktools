#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only Runtime composition for persisted execution history."""

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from ..core import (
    AuthorizationPolicy,
    HmacCursorSigner,
    Page,
    Principal,
    TenantAuthorizationPolicy,
    validate_tenant_id,
)
from ..errors import AIError, ErrorCode
from ..workspace import Workspace
from ._factory import _default_runtime_state, _grant_key
from ._history import StepExecutionHistoryReader
from ._history_service import DefaultExecutionHistoryService
from .service_api import ExecutionHistoryItem, ExecutionTraceItem, TranscriptItem
from .state import RuntimeDomain, RuntimeState


class RuntimeHistory:
    """Stable read-only composition for persisted execution projections."""

    def __init__(
        self,
        service: DefaultExecutionHistoryService,
        *,
        tenant_id: str,
    ) -> None:
        self._service = service
        self._tenant_id = tenant_id

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @classmethod
    def open(
        cls,
        workspace: Workspace,
        *,
        tenant_id: "str | None" = None,
        state: "RuntimeState | None" = None,
        authorization: "AuthorizationPolicy | None" = None,
    ) -> AbstractAsyncContextManager["RuntimeHistory"]:
        return _open_runtime_history(
            workspace,
            tenant_id=tenant_id,
            state=state,
            authorization=authorization,
        )

    async def history(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> Page[ExecutionHistoryItem]:
        return await self._service.history(
            execution_id,
            principal=principal,
            cursor=cursor,
            limit=limit,
        )

    async def trace(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> Page[ExecutionTraceItem]:
        return await self._service.trace(
            execution_id,
            principal=principal,
            cursor=cursor,
            limit=limit,
        )

    async def transcript(
        self,
        execution_id: str,
        *,
        principal: Principal,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> Page[TranscriptItem]:
        return await self._service.transcript(
            execution_id,
            principal=principal,
            cursor=cursor,
            limit=limit,
        )


@asynccontextmanager
async def _open_runtime_history(
    workspace: Workspace,
    *,
    tenant_id: "str | None",
    state: "RuntimeState | None",
    authorization: "AuthorizationPolicy | None",
) -> AsyncIterator[RuntimeHistory]:
    if not isinstance(workspace, Workspace):
        raise TypeError("workspace must be Workspace")
    effective_tenant_id = "default" if tenant_id is None else validate_tenant_id(tenant_id)
    selected_state = state or _default_runtime_state(workspace)
    if not isinstance(selected_state, RuntimeState):
        raise TypeError("state must be RuntimeState")
    initialized = False
    try:
        await selected_state.initialize(
            namespace=workspace.workspace_id,
            tenant_id=effective_tenant_id,
        )
        initialized = True
        if (
            selected_state.namespace != workspace.workspace_id
            or selected_state.tenant_id != effective_tenant_id
        ):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        reader = StepExecutionHistoryReader(
            namespace=workspace.workspace_id,
            executions=selected_state.execution.executions,
            store=selected_state.steps.read_store(RuntimeDomain.EXECUTION),
            cursor_signer=HmacCursorSigner(
                "execution-history",
                _grant_key(workspace),
            ),
            read_model=None,
        )
        service = DefaultExecutionHistoryService(
            selected_state.execution.executions,
            (
                TenantAuthorizationPolicy(effective_tenant_id)
                if authorization is None
                else authorization
            ),
            reader,
        )
        yield RuntimeHistory(service, tenant_id=effective_tenant_id)
    finally:
        if initialized:
            await selected_state.close()


__all__ = ["RuntimeHistory"]
