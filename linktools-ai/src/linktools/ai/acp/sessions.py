#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ACP session ownership and lifecycle state machine."""

import asyncio
import base64
import json
import logging
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping
from uuid import uuid4

from ..execution.domain import RunStatus
from ..governance.identity import PrincipalContext
from ..runtime.facade import Runtime
from .errors import request_error
from .mcp import mcp_descriptor_fingerprint, validate_mcp_descriptors
from .mcp_resources import SessionMcpResources
from .persistence import AcpSessionRecord, AcpSessionRepository
from .session_state import (
    SessionOperationCoordinator,
    SessionOperationKind,
    SessionOperationToken,
    assert_session_invariants,
)
from .session_models import (
    ActiveAcpSession,
    CloseReason,
    SessionCloseFailure,
    SessionCloseResult,
)
from .session_paths import validate_session_paths
from .task_utils import cancel_and_wait, wait_and_observe

logger = logging.getLogger("linktools.ai.acp.sessions")

if TYPE_CHECKING:
    from .client_services import AcpClientServices


class AcpSessionService:
    def __init__(
        self,
        *,
        runtime: Runtime,
        repository: AcpSessionRepository,
        project_root: "str | Path",
        principal: PrincipalContext,
        default_mode_id: str,
        mode_ids: "tuple[str, ...]" = (),
        config_defaults: "Mapping[str, Any] | None" = None,
        client_services: "AcpClientServices | None" = None,
    ) -> None:
        self.runtime = runtime
        self.repository = repository
        self.project_root = Path(os.path.normcase(str(Path(project_root).resolve())))
        self.principal = principal
        self.default_mode_id = default_mode_id
        self.mode_ids = tuple(mode_ids) or (default_mode_id,)
        self.config_defaults = dict(config_defaults or {})
        self.client_services = client_services
        self._active: "dict[str, ActiveAcpSession]" = {}
        self.coordinator = SessionOperationCoordinator()

    @property
    def active_sessions(self) -> Mapping[str, ActiveAcpSession]:
        return self._active

    async def create(
        self,
        *,
        cwd: str,
        additional_directories: "list[str] | None" = None,
        mcp_servers: "list[Any] | None" = None,
    ) -> ActiveAcpSession:
        validate_mcp_descriptors(mcp_servers or ())
        normalized_cwd, directories = validate_session_paths(
            project_root=self.project_root,
            cwd=cwd,
            additional_directories=additional_directories,
        )
        session_id = uuid4().hex
        await self.runtime.create_session(session_id, principal=self.principal)
        now = datetime.now(timezone.utc)
        record = AcpSessionRecord(
            schema_version=1,
            session_id=session_id,
            cwd=normalized_cwd,
            additional_directories=directories,
            mode_id=self.default_mode_id,
            config_values=dict(self.config_defaults),
            mcp_server_fingerprints=tuple(
                sorted(mcp_descriptor_fingerprint(server) for server in (mcp_servers or ()))
            ),
            title=None,
            closed=False,
            created_at=now,
            updated_at=now,
        )
        self.repository.save(record)
        active = ActiveAcpSession(record, asyncio.Lock(), None, SessionMcpResources(tuple(mcp_servers or ())), set(), set())
        self._active[session_id] = active
        return active

    async def get(self, session_id: str) -> ActiveAcpSession:
        active = self._active.get(session_id)
        if active is not None:
            return active
        record = self.repository.load(session_id)
        if record is None:
            raise request_error("unknown_session", session_id=session_id)
        active = ActiveAcpSession(record, asyncio.Lock(), None, SessionMcpResources(), set(), set())
        self._active[session_id] = active
        return active

    async def load_or_resume(
        self,
        *,
        session_id: str,
        cwd: str,
        additional_directories: "list[str] | None",
        mcp_servers: "list[Any] | None",
        replay: bool,
    ) -> ActiveAcpSession:
        active = await self.get(session_id)
        validate_mcp_descriptors(mcp_servers or ())
        normalized_cwd, directories = validate_session_paths(
            project_root=self.project_root,
            cwd=cwd,
            additional_directories=additional_directories,
        )
        operation = await self.coordinator.reserve(
            active,
            SessionOperationKind.LOAD if replay else SessionOperationKind.RESUME,
        )
        new_resources: SessionMcpResources | None = None
        old_descriptors: tuple[Any, ...] = ()
        try:
            async with active.lock:
                if active.record.cwd != normalized_cwd:
                    raise request_error("cwd_mismatch", session_id=session_id)
                fingerprints = tuple(
                    sorted(
                        mcp_descriptor_fingerprint(server)
                        for server in (mcp_servers or ())
                    )
                )
                if fingerprints != active.record.mcp_server_fingerprints:
                    raise request_error("mcp_descriptor_mismatch", session_id=session_id)
                base_record = active.record
                old_mcp_resources = active.mcp_resources
                old_descriptors = old_mcp_resources.descriptors
            if replay:
                await self._assert_complete_history(session_id)
            new_resources = SessionMcpResources(tuple(mcp_servers or ()))
            await old_mcp_resources.close()
            record = replace(
                base_record,
                additional_directories=directories,
                closed=False,
                updated_at=datetime.now(timezone.utc),
            )
            try:
                self.repository.save(record)
            except Exception:
                async with active.lock:
                    active.mcp_resources = SessionMcpResources(old_descriptors)
                raise
            if not await self.coordinator.validate(active, operation):
                raise request_error("session_state_changed", session_id=session_id)
            async with active.lock:
                active.record = record
                active.mcp_resources = new_resources
                new_resources = None
            return active
        except Exception:
            if new_resources is not None:
                await new_resources.close()
            raise
        finally:
            await self.coordinator.release(active, operation)

    async def fork(self, source_session_id: str, *, cwd: str, additional_directories: "list[str] | None", mcp_servers: "list[Any] | None") -> ActiveAcpSession:
        source = await self.get(source_session_id)
        normalized_cwd, directories = validate_session_paths(
            project_root=self.project_root,
            cwd=cwd,
            additional_directories=additional_directories,
        )
        validate_mcp_descriptors(mcp_servers or ())
        fingerprints = tuple(sorted(mcp_descriptor_fingerprint(server) for server in (mcp_servers or ())))
        operation = await self.coordinator.reserve(source, SessionOperationKind.FORK)
        try:
            await self._assert_complete_history(source_session_id)
            async with source.lock:
                source_mode_id = source.record.mode_id
                source_config_values = dict(source.record.config_values)
                source_title = source.record.title
                source_fingerprints = source.record.mcp_server_fingerprints
            if fingerprints != source_fingerprints:
                raise request_error("mcp_descriptor_mismatch", session_id=source_session_id)
            target_id = uuid4().hex
            now = datetime.now(timezone.utc)
            record = AcpSessionRecord(
                schema_version=1,
                session_id=target_id,
                cwd=normalized_cwd,
                additional_directories=directories,
                mode_id=source_mode_id,
                config_values=source_config_values,
                mcp_server_fingerprints=fingerprints,
                title=source_title,
                closed=False,
                created_at=now,
                updated_at=now,
            )
            staged = self.repository.stage(record)
            try:
                await self.runtime.fork_session(source_session_id, target_id, self.principal)
                self.repository.publish(staged, target_id)
            except Exception:
                self.repository.discard(staged)
                raise
            active = ActiveAcpSession(
                record,
                asyncio.Lock(),
                None,
                SessionMcpResources(tuple(mcp_servers or ())),
                set(),
                set(),
            )
            self._active[target_id] = active
            return active
        finally:
            await self.coordinator.release(source, operation)

    async def close(self, session_id: str) -> SessionCloseResult:
        return await self.close_session_resources(session_id, reason="client")

    async def close_session_resources(
        self,
        session_id: str,
        *,
        reason: CloseReason,
    ) -> SessionCloseResult:
        active = await self.get(session_id)
        async with active.lock:
            if (
                active.record.closed
                and active.active_execution_id is None
                and active.operation is None
                and not active.terminal_handles
                and not active.terminal_create_tasks
                and not active.terminal_release_tasks
                and not active.pending_elicitation_ids
                and not active.pending_elicitation_tasks
                and active.pending_permission_task is None
                and active.pending_permission is None
            ):
                return SessionCloseResult(True, ())
        task = await self.coordinator.request_close(
            active,
            lambda: self._run_close_once(active, reason),
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await wait_and_observe(task, timeout=20)
            raise

    async def _run_close_once(
        self,
        active: ActiveAcpSession,
        reason: CloseReason,
    ) -> SessionCloseResult:
        task = asyncio.current_task()
        try:
            return await self._close_once(active, reason)
        finally:
            if task is not None:
                await self.coordinator.clear_close_task(active, task)

    async def cancel_prompt(self, session_id: str) -> None:
        active = await self.get(session_id)
        await self._cancel_prompt(active)

    async def _cancel_prompt(self, active: ActiveAcpSession) -> None:
        async with active.lock:
            operation = active.operation
            if operation is not None and operation.kind is not SessionOperationKind.PROMPT:
                return
            permission_task = active.pending_permission_task
            active.pending_permission_task = None
            active.pending_permission = None
            execution_id = active.active_execution_id
            active.active_execution_id = None
        if permission_task is not None:
            await cancel_and_wait(permission_task, timeout=1)
        if execution_id is not None:
            await asyncio.wait_for(
                self.runtime.cancel(execution_id, principal=self.principal),
                timeout=5,
            )

    async def _close_once(
        self,
        active: ActiveAcpSession,
        reason: CloseReason,
    ) -> SessionCloseResult:
        failures: "list[SessionCloseFailure]" = []
        async with active.lock:
            operation = active.operation
            execution_id = active.active_execution_id
        if operation is not None:
            if operation.kind is SessionOperationKind.PROMPT:
                try:
                    await self._cancel_prompt(active)
                    await asyncio.wait_for(operation.done.wait(), timeout=10)
                except Exception as exc:
                    failures.append(
                        self._close_failure(
                            "execution",
                            execution_id,
                            exc,
                        )
                    )
            else:
                try:
                    await asyncio.wait_for(operation.done.wait(), timeout=20)
                except Exception as exc:
                    failures.append(self._close_failure("operation", None, exc))
        try:
            close_operation = await self.coordinator.reserve_close(active)
        except Exception as exc:
            failures.append(self._close_failure("operation", None, exc))
            return await self._finish_close_failure(active, failures, reason)
        try:
            task_failures = await self._cancel_pending_tasks(active)
            for resource_type, resource_id, error in task_failures:
                failures.append(self._close_failure(resource_type, resource_id, error))
            if self.client_services is not None:
                try:
                    resource_failures = await self.client_services.close_session_resources(active)
                except Exception as exc:
                    resource_failures = (("terminal", None, exc),)
                seen_resources: set[tuple[str, str | None]] = set()
                for resource_type, resource_id, error in resource_failures:
                    key = (resource_type, resource_id)
                    if key in seen_resources:
                        continue
                    seen_resources.add(key)
                    failures.append(self._close_failure(resource_type, resource_id, error))
            async with active.lock:
                mcp_resources = active.mcp_resources
            try:
                await mcp_resources.close()
            except Exception as exc:
                failures.append(self._close_failure("mcp", None, exc))
            close_state_error: BaseException | None = None
            async with active.lock:
                if active.active_execution_id is not None:
                    failures.append(
                        self._close_failure(
                            "execution",
                            active.active_execution_id,
                            RuntimeError("execution is still active"),
                        )
                    )
                if active.pending_permission_task is not None:
                    failures.append(self._close_failure("permission", None, RuntimeError("permission task remains")))
                if active.pending_elicitation_tasks:
                    failures.append(self._close_failure("elicitation", None, RuntimeError("elicitation task remains")))
                if active.terminal_create_tasks:
                    failures.append(self._close_failure("terminal", None, RuntimeError("terminal create task remains")))
                if active.terminal_handles:
                    failures.append(self._close_failure("terminal", None, RuntimeError("terminal resources remain")))
                if active.terminal_release_tasks:
                    failures.append(self._close_failure("terminal", None, RuntimeError("terminal release task remains")))
                if active.pending_elicitation_ids:
                    failures.append(self._close_failure("elicitation", None, RuntimeError("elicitation resources remain")))
                if failures:
                    close_state_error = RuntimeError("session resources remain")
                else:
                    record = replace(active.record, closed=True, updated_at=datetime.now(timezone.utc))
            if close_state_error is not None:
                return await self._finish_close_failure(active, failures, reason, close_operation)
            self.repository.save(record)
            if not await self.coordinator.validate(active, close_operation):
                failures.append(self._close_failure("persistence", None, RuntimeError("session state changed during close")))
                return await self._finish_close_failure(active, failures, reason, close_operation)
            async with active.lock:
                active.record = record
                active.cleanup_required = False
                active.closing_requested = False
            await self.coordinator.release(active, close_operation)
            if logger.isEnabledFor(logging.DEBUG):
                assert_session_invariants(active)
            async with active.lock:
                self._active.pop(active.record.session_id, None)
            logger.info(
                "event=acp.session.closed session_id=%s close_reason=%s pending_task_count=0 terminal_count=0 elicitation_count=0 mcp_state=closed cleanup_required=false",
                active.record.session_id,
                reason,
            )
            return SessionCloseResult(True, ())
        except Exception as exc:
            failures.append(self._close_failure("persistence", None, exc))
            return await self._finish_close_failure(active, failures, reason, close_operation)

    async def _finish_close_failure(
        self,
        active: ActiveAcpSession,
        failures: "list[SessionCloseFailure]",
        reason: CloseReason,
        close_operation: "SessionOperationToken | None" = None,
    ) -> SessionCloseResult:
        async with active.lock:
            active.cleanup_required = True
            active.closing_requested = False
        if close_operation is not None:
            await self.coordinator.release(active, close_operation)
        logger.warning(
            "event=acp.session.cleanup_failed session_id=%s close_reason=%s pending_task_count=%s terminal_count=%s elicitation_count=%s cleanup_required=true error_ids=%s",
            active.record.session_id,
            reason,
            len(active.pending_elicitation_tasks),
            len(active.terminal_handles),
            len(active.pending_elicitation_ids),
            ",".join(item.error_id for item in failures),
        )
        return SessionCloseResult(False, tuple(failures))

    async def _cancel_pending_tasks(
        self,
        active: ActiveAcpSession,
    ) -> "tuple[tuple[str, str | None, BaseException], ...]":
        async with active.lock:
            permission_task = active.pending_permission_task
            permission_token = active.pending_permission
            active.pending_permission_task = None
            active.pending_permission = None
            elicitation_tasks = tuple(active.pending_elicitation_tasks.items())
            create_tasks = tuple(active.terminal_create_tasks)
            release_tasks = tuple(active.terminal_release_tasks.items())
        failures: list[tuple[str, str | None, BaseException]] = []
        if permission_task is not None and not await cancel_and_wait(permission_task, timeout=1):
            async with active.lock:
                if active.pending_permission_task is None:
                    active.pending_permission_task = permission_task
                    active.pending_permission = permission_token
            failures.append(("permission", None, TimeoutError("permission task ignored cancellation")))
        for task_id, task in elicitation_tasks:
            if not await cancel_and_wait(task, timeout=10):
                failures.append(("elicitation", task_id, TimeoutError("elicitation task ignored cancellation")))
        for task in create_tasks:
            if not await cancel_and_wait(task, timeout=5):
                failures.append(("terminal", None, TimeoutError("terminal create task ignored cancellation")))
        for terminal_id, task in release_tasks:
            if not await cancel_and_wait(task, timeout=5):
                failures.append(("terminal", terminal_id, TimeoutError("terminal release task ignored cancellation")))
        return tuple(failures)

    @staticmethod
    def _close_failure(resource_type: str, resource_id: "str | None", error: BaseException) -> SessionCloseFailure:
        return SessionCloseFailure(resource_type, resource_id, uuid4().hex)

    async def set_mode(self, session_id: str, mode_id: str) -> ActiveAcpSession:
        active = await self.get(session_id)
        operation = await self.coordinator.reserve(active, SessionOperationKind.SET_MODE)
        try:
            if mode_id not in self.mode_ids:
                raise request_error("unknown_mode", session_id=session_id)
            async with active.lock:
                record = replace(active.record, mode_id=mode_id, updated_at=datetime.now(timezone.utc))
            self.repository.save(record)
            if not await self.coordinator.validate(active, operation):
                raise request_error("session_state_changed", session_id=session_id)
            async with active.lock:
                active.record = record
                return active
        finally:
            await self.coordinator.release(active, operation)

    async def set_config(self, session_id: str, config_id: str, value: Any) -> ActiveAcpSession:
        active = await self.get(session_id)
        operation = await self.coordinator.reserve(active, SessionOperationKind.SET_CONFIG)
        try:
            if config_id not in self.config_defaults:
                raise request_error("unknown_config_option", session_id=session_id)
            if not isinstance(value, type(self.config_defaults[config_id])):
                raise request_error("invalid_config_value", session_id=session_id)
            async with active.lock:
                values = dict(active.record.config_values)
                values[config_id] = value
                record = replace(active.record, config_values=values, updated_at=datetime.now(timezone.utc))
            self.repository.save(record)
            if not await self.coordinator.validate(active, operation):
                raise request_error("session_state_changed", session_id=session_id)
            async with active.lock:
                active.record = record
                return active
        finally:
            await self.coordinator.release(active, operation)

    async def list(self, *, cwd: "str | None" = None, cursor: "str | None" = None) -> "tuple[tuple[AcpSessionRecord, ...], str | None]":
        records = list(self.repository.list())
        if cwd is not None:
            normalized, _ = validate_session_paths(project_root=self.project_root, cwd=cwd, additional_directories=[])
            records = [record for record in records if record.cwd == normalized]
        records.sort(key=lambda record: (-record.updated_at.timestamp(), record.session_id))
        offset = self._decode_cursor(records, cursor)
        page = records[offset: offset + 50]
        next_cursor = self._encode_cursor(page[-1]) if offset + 50 < len(records) else None
        return tuple(page), next_cursor

    async def _assert_complete_history(self, session_id: str) -> None:
        views = await self.runtime.get_session_messages(session_id=session_id, principal=self.principal)
        for view in views:
            capture_state = getattr(view, "capture_state", None)
            if getattr(view, "status", None) is not RunStatus.COMPLETED or getattr(capture_state, "value", capture_state) != "complete":
                raise request_error("incomplete_history", session_id=session_id)

    @staticmethod
    def _encode_cursor(record: AcpSessionRecord) -> str:
        raw = json.dumps(
            {
                "v": 1,
                "updated_at": record.updated_at.isoformat().replace("+00:00", "Z"),
                "session_id": record.session_id,
            },
            separators=(",", ":"),
        ).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    @staticmethod
    def _decode_cursor(records: "list[AcpSessionRecord]", cursor: "str | None") -> int:
        if cursor is None:
            return 0
        try:
            raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
            value = json.loads(raw)
            if value.get("v") != 1 or not isinstance(value.get("updated_at"), str) or not isinstance(value.get("session_id"), str):
                raise ValueError
            updated_at = datetime.fromisoformat(value["updated_at"].replace("Z", "+00:00"))
            for index, record in enumerate(records):
                if record.updated_at == updated_at and record.session_id == value["session_id"]:
                    return index + 1
            raise ValueError("cursor record is no longer available")
        except (ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise request_error("invalid_cursor") from exc


__all__ = [
    "AcpSessionService",
]
