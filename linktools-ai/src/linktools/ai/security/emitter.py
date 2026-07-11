"""Shared event emission policy for security and observability events."""

import logging
from typing import Any, Protocol, runtime_checkable

from ..errors import ToolSecurityAuditError

_LOGGER = logging.getLogger(__name__)


@runtime_checkable
class SecurityEventEmitter(Protocol):
    async def emit_security(self, event: Any) -> None: ...
    async def emit_observability(self, event: Any) -> None: ...


class EventStoreSecurityEventEmitter:
    def __init__(self, event_store: Any, *, context: Any = None,
                 failure_mode: Any = "fail_closed") -> None:
        self._store = event_store
        self._context = context
        self._failure_mode = getattr(failure_mode, "value", failure_mode)

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
                payload=event,
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
