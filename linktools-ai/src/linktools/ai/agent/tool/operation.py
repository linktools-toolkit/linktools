"""State for approved tool execution only."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ...storage.coordination.lease import Lease
from ...json import JsonValue


class ToolOperationStatus(str, Enum):
    PREPARED = "prepared"
    CLAIMED = "claimed"
    COMPLETED = "completed"
    FAILED = "failed"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True, slots=True)
class ToolOperation:
    id: str
    tenant_id: str | None
    execution_id: str
    tool_call_id: str
    idempotency_key: str
    tool_name: str
    arguments_hash: str
    binding_fingerprint: str
    status: ToolOperationStatus
    replay_safe: bool = False
    lease: Lease = Lease()
    result: JsonValue | None = None
    error: JsonValue | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def owner(self) -> str | None:
        return self.lease.owner

    @property
    def fence(self) -> int:
        return self.lease.fence

    @property
    def lease_expires_at(self) -> datetime | None:
        return self.lease.expires_at
