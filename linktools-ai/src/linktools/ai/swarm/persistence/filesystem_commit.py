#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FilesystemSwarmCommitCoordinator: commit-id-keyed idempotent commit for
the swarm + swarm-step lifecycle points on Filesystem storage.

Mirrors the SQL reference impl's contract: every operation is idempotent by
commit_id (a retried call with the SAME (commit_id, request_hash) returns the
recorded result; same id + different payload raises SwarmCommitConflictError),
and the swarm-state write is journaled so a crash mid-commit is reconciled
at next startup.

Filesystem swarm commits are simpler than run commits (no critical events to
coordinate -- the driving Run's events are the RunCoordinator's concern), so
a single-shot journal suffices rather than a multi-state state machine: the
in-flight journal is written before the swarm_store op and removed after the
completion log lands. ``recover_incomplete_commits`` runs at startup and
marks the swarm_run of any surviving in-flight journal FAILED (fail-closed)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from ..commit import SwarmCommitConflictError
from ...run.persistence._replay import canonical_request_hash

if TYPE_CHECKING:
    from .filesystem import FilesystemSwarmStore
    from ..commit import (
        CancelSwarmCommand,
        CompleteSwarmCommand,
        CompleteSwarmStepCommand,
        FailSwarmCommand,
        FailSwarmStepCommand,
        StartSwarmCommand,
        StartSwarmStepCommand,
    )

_LOGGER = logging.getLogger(__name__)


def _hash_segment(commit_id: str) -> str:
    return hashlib.sha256(commit_id.encode("utf-8")).hexdigest()


class FilesystemSwarmCommitCoordinator:
    """Filesystem reference implementation of SwarmCommitCoordinator. Uses a
    per-coordinator completion log under ``{transactions_root}/swarm_completed/``
    so retries return the originally-recorded result, plus an in-flight
    journal under ``{transactions_root}/swarm_inflight/`` so a crash mid-
    commit is reconciled at next startup."""

    def __init__(
        self,
        swarm_store: "FilesystemSwarmStore",
        *,
        transactions_root: "str | Path",
    ) -> None:
        self._swarm_store = swarm_store
        root = Path(transactions_root)
        self._completed_dir = root / "swarm_completed"
        self._inflight_dir = root / "swarm_inflight"
        self._completed_dir.mkdir(parents=True, exist_ok=True)
        self._inflight_dir.mkdir(parents=True, exist_ok=True)

    async def recover_incomplete_commits(self) -> None:
        """For each in-flight journal left by a crash: mark the affected
        swarm_run FAILED (fail-closed), then delete the journal. Best-effort
        -- a swarm_run that's already terminal (or gone) is tolerated, the
        journal is still cleared so the next start is clean. Runs at Runtime
        startup before the Runtime accepts new requests."""
        from ..models import SwarmStatus

        for path in sorted(self._inflight_dir.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                swarm_run_id = raw.get("swarm_run_id")
                if swarm_run_id:
                    try:
                        current = await self._swarm_store.get_run(swarm_run_id)
                        if current is not None and current.status in (
                            SwarmStatus.PENDING,
                            SwarmStatus.RUNNING,
                            SwarmStatus.CANCELLING,
                        ):
                            await self._swarm_store.update_run(
                                swarm_run_id,
                                expected_version=current.version,
                                status=SwarmStatus.FAILED,
                            )
                    except Exception:  # noqa: BLE001
                        _LOGGER.warning(
                            "swarm recovery could not mark run %s FAILED for "
                            "in-flight journal %s",
                            swarm_run_id,
                            path.name,
                        )
            except (OSError, ValueError):
                _LOGGER.warning(
                    "swarm recovery could not parse in-flight journal %s",
                    path.name,
                )
            try:
                path.unlink()
            except OSError:
                pass

    def _read_completion(
        self, commit_id: str, request_hash: bytes
    ) -> "Mapping[str, Any] | None":
        path = self._completed_dir / f"{_hash_segment(commit_id)}.json"
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        recorded = bytes.fromhex(raw.get("request_hash", ""))
        if recorded != request_hash:
            raise SwarmCommitConflictError(
                f"swarm commit {commit_id!r} replayed with a different request"
            )
        return raw.get("result") or {}

    def _write_completion(
        self,
        commit_id: str,
        request_hash: bytes,
        result: Mapping[str, Any],
    ) -> None:
        path = self._completed_dir / f"{_hash_segment(commit_id)}.json"
        payload = {
            "commit_id": commit_id,
            "request_hash": request_hash.hex(),
            "result": dict(result),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        fd = os.open(tmp, os.O_WRONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)

    def _write_inflight(
        self,
        commit_id: str,
        operation: str,
        swarm_run_id: str,
        request_hash: bytes,
    ) -> None:
        path = self._inflight_dir / f"{_hash_segment(commit_id)}.json"
        payload = {
            "commit_id": commit_id,
            "operation": operation,
            "swarm_run_id": swarm_run_id,
            "request_hash": request_hash.hex(),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        fd = os.open(tmp, os.O_WRONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)

    def _clear_inflight(self, commit_id: str) -> None:
        path = self._inflight_dir / f"{_hash_segment(commit_id)}.json"
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def _check_replay(
        self, commit_id: str, operation: str, request_payload: Mapping[str, Any]
    ) -> "Mapping[str, Any] | None":
        return self._read_completion(
            commit_id, canonical_request_hash(operation, request_payload)
        )

    def _record(
        self,
        commit_id: str,
        operation: str,
        swarm_run_id: str,
        request_payload: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> None:
        request_hash = canonical_request_hash(operation, request_payload)
        self._write_completion(commit_id, request_hash, result)
        self._clear_inflight(commit_id)

    async def _run_commit(
        self,
        operation: str,
        command: Any,
        request_payload: Mapping[str, Any],
        run_business: Any,
    ) -> Any:
        """Wrap a business-write closure with replay detection + inflight
        journaling + completion-log recording. ``run_business`` is a sync or
        async callable that performs the actual swarm_store op and returns
        the result dict to record."""
        commit_id = command.commit_id
        swarm_run_id = command.swarm_run_id
        replay = self._check_replay(commit_id, operation, request_payload)
        if replay is not None:
            return replay
        # Journal the intent BEFORE the business write so a crash mid-write
        # is reconcilable (recovery marks the swarm_run FAILED).
        self._write_inflight(
            commit_id,
            operation,
            swarm_run_id,
            canonical_request_hash(operation, request_payload),
        )
        result = await run_business()
        self._record(commit_id, operation, swarm_run_id, request_payload, result)
        return result

    async def start(self, command: "StartSwarmCommand") -> Any:
        request_payload = {
            "swarm_run_id": command.swarm_run_id,
            "expected_version": command.expected_version,
            "new_run_id": str(command.payload.get("id", "")),
            "driving_run_id": str(command.payload.get("run_id", "")),
        }

        async def _business():
            from ..models import SwarmRun

            run = SwarmRun(**dict(command.payload))
            created = await self._swarm_store.create_run(run)
            return {"swarm_run_id": created.id, "version": created.version}

        return await self._run_commit("start", command, request_payload, _business)

    async def start_step(self, command: "StartSwarmStepCommand") -> Any:
        request_payload = {
            "swarm_run_id": command.swarm_run_id,
            "step_attempt_id": command.step_attempt_id,
            "expected_version": command.expected_version,
            "new_task_id": str(command.payload.get("id", "")),
        }

        async def _business():
            from ..models import SwarmStep

            step = SwarmStep(**dict(command.payload))
            created = await self._swarm_store.create_task(step)
            return {"task_id": created.id, "version": created.version}

        return await self._run_commit(
            "start_step", command, request_payload, _business
        )

    async def complete_step(self, command: "CompleteSwarmStepCommand") -> Any:
        request_payload = {
            "swarm_run_id": command.swarm_run_id,
            "step_attempt_id": command.step_attempt_id,
            "expected_version": command.expected_version,
            "task_id": str(command.payload.get("task_id", "")),
            "active_run_id": str(command.payload.get("active_run_id", "")),
        }

        async def _business():
            from ...run.models import RunResult

            task_id = str(command.payload.get("task_id", ""))
            result = RunResult(**dict(command.payload.get("result", {})))
            updated = await self._swarm_store.complete_task(
                task_id,
                result,
                expected_version=command.expected_version,
                active_run_id=command.payload.get("active_run_id"),
            )
            return {"task_id": updated.id, "version": updated.version}

        return await self._run_commit(
            "complete_step", command, request_payload, _business
        )

    async def fail_step(self, command: "FailSwarmStepCommand") -> Any:
        request_payload = {
            "swarm_run_id": command.swarm_run_id,
            "step_attempt_id": command.step_attempt_id,
            "expected_version": command.expected_version,
            "task_id": str(command.payload.get("task_id", "")),
            "active_run_id": str(command.payload.get("active_run_id", "")),
            "error_type": str(
                dict(command.payload.get("error", {})).get("error_type", "")
            ),
        }

        async def _business():
            from ...run.models import RunErrorInfo

            task_id = str(command.payload.get("task_id", ""))
            error = RunErrorInfo(**dict(command.payload.get("error", {})))
            updated = await self._swarm_store.fail_task(
                task_id,
                error,
                expected_version=command.expected_version,
                active_run_id=command.payload.get("active_run_id"),
            )
            return {"task_id": updated.id, "version": updated.version}

        return await self._run_commit(
            "fail_step", command, request_payload, _business
        )

    async def complete(self, command: "CompleteSwarmCommand") -> Any:
        request_payload = {
            "swarm_run_id": command.swarm_run_id,
            "expected_version": command.expected_version,
        }

        async def _business():
            from ..models import SwarmStatus

            updated = await self._swarm_store.update_run(
                command.swarm_run_id,
                expected_version=command.expected_version,
                status=SwarmStatus.SUCCEEDED,
            )
            return {"swarm_run_id": updated.id, "version": updated.version}

        return await self._run_commit(
            "complete", command, request_payload, _business
        )

    async def fail(self, command: "FailSwarmCommand") -> Any:
        request_payload = {
            "swarm_run_id": command.swarm_run_id,
            "expected_version": command.expected_version,
            "error_type": str(
                dict(command.payload.get("error", {})).get("error_type", "")
            ),
        }

        async def _business():
            from ..models import SwarmStatus

            updated = await self._swarm_store.update_run(
                command.swarm_run_id,
                expected_version=command.expected_version,
                status=SwarmStatus.FAILED,
            )
            return {"swarm_run_id": updated.id, "version": updated.version}

        return await self._run_commit(
            "fail", command, request_payload, _business
        )

    async def cancel(self, command: "CancelSwarmCommand") -> Any:
        request_payload = {
            "swarm_run_id": command.swarm_run_id,
            "expected_version": command.expected_version,
            "reason": str(command.payload.get("reason", "")),
        }

        async def _business():
            from ..models import SwarmStatus

            updated = await self._swarm_store.update_run(
                command.swarm_run_id,
                expected_version=command.expected_version,
                status=SwarmStatus.CANCELLED,
            )
            return {"swarm_run_id": updated.id, "version": updated.version}

        return await self._run_commit(
            "cancel", command, request_payload, _business
        )


__all__: "list[str]" = ["FilesystemSwarmCommitCoordinator"]
