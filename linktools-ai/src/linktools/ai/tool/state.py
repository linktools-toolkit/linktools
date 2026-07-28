"""State for approved tool execution only."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib
import json
from typing import Any


def compute_arguments_hash(arguments: Any) -> str:
    return hashlib.sha256(json.dumps(arguments, sort_keys=True, default=str).encode()).hexdigest()


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
    owner: str | None = None
    fence: int = 0
    lease_expires_at: datetime | None = None
    result: Any = None
    error: Any = None
