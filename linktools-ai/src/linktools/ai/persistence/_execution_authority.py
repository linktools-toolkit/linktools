#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact same-row SQL owner for coordinated Execution authority."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from ..errors import AIError, ErrorCode
from ..storage import TurnScopeIdentity
from ._record_owner import EXECUTION_SHELL_KIND
from ._turn_sql import turn_hex, turn_session_address
from ._turn_sql_schema import RUNTIME_TURN_SCOPE
from .state import record_key

if TYPE_CHECKING:
    from sqlalchemy import Table
    from sqlalchemy.sql import ColumnElement


class SourceTurnRevoked(AIError):
    """The exact persisted runtime authority can no longer authorize writes."""

    def __init__(self) -> None:
        super().__init__(ErrorCode.STORAGE_OWNER_MISMATCH)


@dataclass(frozen=True, slots=True)
class ExecutionAuthority:
    scope: TurnScopeIdentity
    execution_id: str
    slot: str
    token: str
    fence: int

    @property
    def key(self) -> bytes:
        return record_key(self.scope.namespace, self.scope.tenant_id, "execution", "execution", (self.execution_id,))

    def authority_values(self) -> dict[str, str | int | bytes]:
        if self.scope.scope != RUNTIME_TURN_SCOPE or type(self.fence) is not int or self.fence <= 0:
            raise ValueError("invalid source execution authority")
        kind, slot = turn_session_address(self.slot)
        return dict(turn_slot_kind=kind, turn_slot_key=slot,
                    turn_token=turn_hex(self.token, 32), turn_fence=self.fence)

    def shell_values(self, *, store_key: bytes, expiry: datetime) -> dict[str, object]:
        from hashlib import sha256

        key = self.key
        if expiry.tzinfo is None:
            raise ValueError("execution authority expiry must be timezone-aware")
        return dict(store_key=store_key, key=key, kind=EXECUTION_SHELL_KIND, sort_key=key, sort_prefix=key[:128],
                    scan_kind="k:" + EXECUTION_SHELL_KIND, storage_revision=0, payload_json={},
                    key_locator=sha256(key).hexdigest(), turn_expiry=expiry, **self.authority_values())

    def owner_matches(self, table: "Table", store_key: bytes) -> "ColumnElement[bool]":
        from hashlib import sha256
        from sqlalchemy import and_

        key = self.key
        return and_(table.c.store_key == store_key, table.c.key_locator == sha256(key).hexdigest(), table.c.key == key,
                    *(table.c[name] == value for name, value in self.authority_values().items()))


@dataclass(slots=True)
class InitialExecutionAuthority:
    """The exact source and expiry installed by grouped publication."""

    proof: ExecutionAuthority
    expiry: datetime | timedelta
    control_id: int | None = None


__all__ = ["ExecutionAuthority", "SourceTurnRevoked"]
