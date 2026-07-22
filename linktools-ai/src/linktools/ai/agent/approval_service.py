"""Principal-bound approval operations."""

from typing import Any

from ..errors import PrincipalAccessDeniedError
from ..identity.principal import PrincipalContext


class ApprovalService:
    def __init__(self, store: Any, authorization: Any = None) -> None:
        self._store = store
        self._authorization = authorization

    async def approve(self, request_id: str, *, principal: PrincipalContext, expected_version: int):
        request = await self._store.get(request_id)
        if request is None:
            raise PrincipalAccessDeniedError("approval does not exist")
        if self._authorization is not None:
            from ..governance.security.authorization import AuthorizationTarget
            from ..governance.security.actions import SecurityAction
            await self._authorization.authorize(
                principal, SecurityAction.APPROVAL_APPROVE,
                AuthorizationTarget(kind="approval", id=request.id, tenant_id=request.tenant_id),
            )
        else:
            principal.require_tenant(request.tenant_id)
        return await self._store.approve(
            request_id,
            expected_version=expected_version,
            resolved_by=principal.resolved_by,
        )

    async def reject(self, request_id: str, *, principal: PrincipalContext, expected_version: int, reason: str | None = None):
        request = await self._store.get(request_id)
        if request is None:
            raise PrincipalAccessDeniedError("approval does not exist")
        if self._authorization is not None:
            from ..governance.security.authorization import AuthorizationTarget
            from ..governance.security.actions import SecurityAction
            await self._authorization.authorize(
                principal, SecurityAction.APPROVAL_REJECT,
                AuthorizationTarget(kind="approval", id=request.id, tenant_id=request.tenant_id),
            )
        else:
            principal.require_tenant(request.tenant_id)
        return await self._store.reject(
            request_id,
            expected_version=expected_version,
            resolved_by=principal.resolved_by,
            reason=reason,
        )
