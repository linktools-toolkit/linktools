"""Shared event emission policy for security and observability events."""

import logging
import dataclasses
import json
import re
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

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
    _SECRET_KEYS = {"authorization", "api_key", "apikey", "token", "access_token",
                    "refresh_token", "password", "secret", "cookie", "set-cookie"}
    _MAX_TEXT = 4096
    _MAX_PAYLOAD = 32768

    def sanitize(self, event: Any) -> Any:
        result = self._value(event)
        try:
            serialized = json.dumps(result, default=str, ensure_ascii=False)
            if len(serialized.encode("utf-8")) > self._MAX_PAYLOAD:
                result = {"event_type": type(event).__name__,
                          "payload": "<redacted: payload exceeded limit>"}
        except Exception:
            pass
        return result

    def _value(self, value: Any, key: str | None = None) -> Any:
        if key and key.lower() in self._SECRET_KEYS:
            return "***REDACTED***"
        if isinstance(value, str):
            value = redact_exception(RuntimeError(value), max_chars=self._MAX_TEXT)
            return re.sub(
                r"([?&](?:token|api_key|apikey|access_token|secret|password)=)[^&\s]+",
                r"\1***REDACTED***", value, flags=re.IGNORECASE)
        if isinstance(value, Mapping):
            return {str(k): self._value(v, str(k)) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return type(value)(self._value(v) for v in value)
        if dataclasses.is_dataclass(value):
            values = {f.name: self._value(getattr(value, f.name), f.name)
                      for f in dataclasses.fields(value)}
            try:
                return type(value)(**values)
            except Exception:
                return values
        return value

    def _truncate(self, value: Any) -> Any:
        if isinstance(value, str):
            return value[: self._MAX_TEXT] + "<truncated>"
        if isinstance(value, Mapping):
            return {k: self._truncate(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self._truncate(v) for v in value)
        return value


class EventStoreSecurityEventEmitter:
    def __init__(self, event_store: Any, *, context: Any = None,
                 failure_mode: Any = "fail_closed",
                 sanitizer: SecurityEventSanitizer | None = None) -> None:
        self._store = event_store
        self._context = context
        self._failure_mode = getattr(failure_mode, "value", failure_mode)
        self._sanitizer = sanitizer or DefaultSecurityEventSanitizer()

    async def _append(self, event: Any) -> None:
        if self._store is None:
            return
        ctx = self._context
        run_id = getattr(ctx, "run_id", None) if ctx else None
        try:
            await self._store.append(
                stream_id=run_id or "", run_id=run_id,
                root_run_id=(getattr(ctx, "root_run_id", None) or run_id) if ctx else run_id,
                parent_run_id=getattr(ctx, "parent_run_id", None) if ctx else None,
                session_id=getattr(ctx, "session_id", None) if ctx else None,
                runnable_id=getattr(ctx, "runnable_id", None) if ctx else None,
                payload=self._sanitizer.sanitize(event),
            )
        except Exception as exc:
            raise ToolSecurityAuditError(
                f"failed to persist security event {type(event).__name__}") from exc

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
