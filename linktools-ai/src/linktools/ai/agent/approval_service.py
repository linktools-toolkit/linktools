"""Principal-bound approval operations."""

from typing import Any

from ..errors import PrincipalAccessDeniedError
from ..security.principal import PrincipalContext


class ApprovalService:
    def __init__(self, store: Any) -> None:
        self._store = store

    async def approve(self, request_id: str, *, principal: PrincipalContext, expected_version: int):
        request = await self._store.get(request_id)
        if request is None:
            raise PrincipalAccessDeniedError("approval does not exist")
        if request.tenant_id is not None:
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
        if request.tenant_id is not None:
            principal.require_tenant(request.tenant_id)
        return await self._store.reject(
            request_id,
            expected_version=expected_version,
            resolved_by=principal.resolved_by,
            reason=reason,
        )
