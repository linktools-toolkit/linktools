"""Shared event emission policy for security and observability events."""

import logging
import dataclasses
import json
import re
from collections.abc import Mapping
from datetime import date, datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from ..errors import ToolSecurityAuditError
from .redact import redact_exception

_LOGGER = logging.getLogger(__name__)


@runtime_checkable
class SecurityEventEmitter(Protocol):
    async def emit_security(self, event: Any) -> None: ...
    async def emit_observability(self, event: Any) -> None: ...


class SecurityEventSanitizer(Protocol):
    def sanitize(self, event: Any) -> Any: ...


class DefaultSecurityEventSanitizer:
    _SECRET_KEYS = {
        "authorization",
        "api_key",
        "apikey",
        "token",
        "access_token",
        "refresh_token",
        "password",
        "secret",
        "cookie",
        "set-cookie",
    }
    _MAX_TEXT = 4096
    _MAX_PAYLOAD = 32768

    def sanitize(self, event: Any) -> Any:
        result = self._value(event)
        try:
            serialized = json.dumps(result, default=str, ensure_ascii=False)
            size = len(serialized.encode("utf-8"))
        except Exception:
            # If the sanitized value cannot be sized it cannot overflow either;
            # return it as-is rather than synthesizing a placeholder.
            return result
        if size > self._MAX_PAYLOAD:
            # Return a valid EventPayload dataclass (not a dict): FileEventStore
            # persists via dataclasses.asdict and reconstructs by class name, so a
            # dict here would TypeError and break the security audit trail.
            from ..events.payloads import TruncatedSecurityEvent

            return TruncatedSecurityEvent(
                original_event_type=type(event).__name__,
                reason="payload_too_large",
                original_size_bytes=size,
            )
        return result

    def _value(self, value: Any, key: str | None = None) -> Any:
        if key and key.lower() in self._SECRET_KEYS:
            return "***REDACTED***"
        if isinstance(value, str):
            value = redact_exception(RuntimeError(value), max_chars=self._MAX_TEXT)
            return re.sub(
                r"([?&](?:token|api_key|apikey|access_token|secret|password)=)[^&\s]+",
                r"\1***REDACTED***",
                value,
                flags=re.IGNORECASE,
            )
        if isinstance(value, Mapping):
            return {str(k): self._value(v, str(k)) for k, v in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            # Every sequence/set container normalizes to a JSON array so the
            # sanitized event is JSON-serializable (sets and tuples are not).
            return [self._value(v) for v in value]
        if isinstance(value, (bytes, bytearray)):
            return "<binary-redacted>"
        if isinstance(value, Enum):
            return self._value(value.value)
        if isinstance(value, UUID):
            return str(value)
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if dataclasses.is_dataclass(value):
            values = {
                f.name: self._value(getattr(value, f.name), f.name)
                for f in dataclasses.fields(value)
            }
            try:
                return type(value)(**values)
            except Exception:
                return values
        if value is None or isinstance(value, (bool, int, float)):
            return value
        # Unknown object: coerce to a truncated safe string so the sanitized
        # event is always JSON-serializable without relying on the store's
        # default=str fallback.
        text = str(value)
        if len(text) > self._MAX_TEXT:
            return text[: self._MAX_TEXT] + "<truncated>"
        return text

    def _truncate(self, value: Any) -> Any:
        if isinstance(value, str):
            return value[: self._MAX_TEXT] + "<truncated>"
        if isinstance(value, Mapping):
            return {k: self._truncate(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self._truncate(v) for v in value)
        return value


class EventStoreSecurityEventEmitter:
    def __init__(
        self,
        event_store: Any,
        *,
        context: Any = None,
        failure_mode: Any = "fail_closed",
        sanitizer: SecurityEventSanitizer | None = None,
    ) -> None:
        self._store = event_store
        self._context = context
        self._failure_mode = getattr(failure_mode, "value", failure_mode)
        self._sanitizer = sanitizer or DefaultSecurityEventSanitizer()

    async def _append(self, event: Any) -> None:
        if self._store is None:
            return
        ctx = self._context
        run_id = getattr(ctx, "run_id", None) if ctx else None
        from ..events.context import EventContext, append_event

        try:
            await append_event(
                self._store,
                EventContext(
                    stream_id=run_id or "",
                    run_id=run_id,
                    root_run_id=(getattr(ctx, "root_run_id", None) or run_id)
                    if ctx
                    else run_id,
                    parent_run_id=getattr(ctx, "parent_run_id", None) if ctx else None,
                    session_id=getattr(ctx, "session_id", None) if ctx else None,
                    runnable_id=getattr(ctx, "runnable_id", None) if ctx else None,
                ),
                self._sanitizer.sanitize(event),
            )
        except Exception as exc:
            raise ToolSecurityAuditError(
                f"failed to persist security event {type(event).__name__}"
            ) from exc

    async def emit_security(self, event: Any) -> None:
        try:
            await self._append(event)
        except ToolSecurityAuditError:
            if self._failure_mode != "best_effort":
                raise
            _LOGGER.warning("security event emission failed in best-effort mode")

    async def emit_observability(self, event: Any) -> None:
        try:
            await self._append(event)
        except ToolSecurityAuditError:
            _LOGGER.debug("observability event emission failed", exc_info=True)


class CollectingSecurityEventEmitter:
    """A SecurityEventEmitter that records sanitized events in memory instead of
    persisting them. Used by Runtime.inspect so a capability resolution that
    emits SecurityDegraded (e.g. MCP best-effort discovery) behaves the same
    under inspection as under a real run -- the degradation is observable, not
    silently swallowed or turned into a hard failure.

    Nothing is written to an EventStore: inspection has no run_id and must not
    produce audit side effects. The sanitizer still runs so any secret in an
    event field is redacted before the event is exposed via the warnings API."""

    def __init__(self, *, sanitizer: "SecurityEventSanitizer | Any") -> None:
        self._sanitizer = sanitizer
        self._security_events: "list[Any]" = []
        self._observability_events: "list[Any]" = []

    async def emit_security(self, event: Any) -> None:
        self._security_events.append(self._sanitizer.sanitize(event))

    async def emit_observability(self, event: Any) -> None:
        self._observability_events.append(self._sanitizer.sanitize(event))

    @property
    def security_events(self) -> "tuple[Any, ...]":
        return tuple(self._security_events)

    @property
    def observability_events(self) -> "tuple[Any, ...]":
        return tuple(self._observability_events)
