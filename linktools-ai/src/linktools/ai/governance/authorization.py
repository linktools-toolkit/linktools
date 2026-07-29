"""Authorization boundary for runtime execution operations."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from ..errors import PrincipalAccessDeniedError
from .identity import PrincipalContext

class OwnedSession(Protocol):
    tenant_id: str | None
    user_id: str | None


class ExecutionAction(StrEnum):
    RUN = "run"
    RESUME = "resume"
    CANCEL = "cancel"
    DECIDE_APPROVAL = "decide_approval"
    INSPECT = "inspect"


class AuthorizationPolicy(Protocol):
    def assert_execution_access(
        self,
        *,
        principal: PrincipalContext,
        tenant_id: str | None,
        user_id: str | None,
        action: ExecutionAction,
    ) -> None: ...

    def assert_session_access(
        self,
        *,
        principal: PrincipalContext,
        session: OwnedSession,
    ) -> None: ...


class OwnershipAuthorizationPolicy:
    """Fail-closed tenant/user ownership with action scopes.

    Unrestricted principals bypass the scope check, but never ownership.
    """

    def assert_execution_access(
        self,
        *,
        principal: PrincipalContext,
        tenant_id: str | None,
        user_id: str | None,
        action: ExecutionAction,
    ) -> None:
        principal.require_tenant(tenant_id)
        if user_id is not None and principal.user_id != user_id:
            raise PrincipalAccessDeniedError("record is not owned by this user")
        if not (
            principal.scopes.unrestricted
            or principal.scopes.contains(f"execution:{action.value}")
        ):
            raise PrincipalAccessDeniedError(
                f"principal lacks execution:{action.value} scope"
            )

    def assert_session_access(
        self,
        *,
        principal: PrincipalContext,
        session: OwnedSession,
    ) -> None:
        self.assert_execution_access(
            principal=principal,
            tenant_id=session.tenant_id,
            user_id=session.user_id,
            action=ExecutionAction.RUN,
        )


__all__ = [
    "AuthorizationPolicy",
    "ExecutionAction",
    "OwnershipAuthorizationPolicy",
]
