"""State for approved tool execution only."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib
import json
from typing import Any

from ..storage.coordination.lease import Lease
from ..storage.json import canonical_json_bytes


def compute_arguments_hash(arguments: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(arguments)).hexdigest()


class ToolOperationStatus(str, Enum):
    PREPARED = "prepared"
    CLAIMED = "claimed"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ToolOperation:
    id: str
    tenant_id: str | None
    run_id: str
    tool_call_id: str
    idempotency_key: str
    tool_name: str
    arguments_hash: str
    status: ToolOperationStatus
    lease: Lease = Lease()
    result: Any = None
    error: Any = None
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
