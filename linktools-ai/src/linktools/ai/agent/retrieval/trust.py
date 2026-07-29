"""Trust metadata for retrieved context; retrieved text is never trusted instructions."""

from dataclasses import dataclass
from enum import Enum


class ContextTrustLevel(str, Enum):
    SYSTEM = "system"
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"


@dataclass(frozen=True, slots=True)
class ContextItem:
    content: str
    source_id: str
    tenant_id: str
    revision: str | None = None
    sha256: str | None = None
    trust_level: ContextTrustLevel = ContextTrustLevel.UNTRUSTED
