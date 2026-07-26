#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FilesystemSwarmCommitCoordinator: commit-id-keyed idempotent commit for
the swarm + swarm-step lifecycle points on Filesystem storage.

Mirrors the SQL reference impl's contract: every operation is idempotent by
commit_id (a retried call with the SAME (commit_id, request_hash) returns the
recorded result; same id + different payload raises SwarmCommitConflictError),
and the swarm-state write is journaled so a crash mid-commit is reconciled
at next startup.

Filesystem swarm commits use a single-shot journal for the state/event/log
boundary rather than a multi-state state machine: the
in-flight journal is written before the swarm_store op and removed after the
completion log lands. ``recover_incomplete_commits`` runs at startup and
marks the swarm_run of any surviving in-flight journal FAILED (fail-closed)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import base64
import binascii
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from ..commit import SwarmCommitConflictError, SwarmCommitPolicy
from ...run.persistence._replay import canonical_request_hash
from .codec import SwarmCommitCodec

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
        SwarmCommitResult,
    )

_LOGGER = logging.getLogger(__name__)


def _hash_segment(commit_id: str) -> str:
    return hashlib.sha256(commit_id.encode("utf-8")).hexdigest()


def _fsync_parent(path: Path) -> None:
    """Persist a directory entry update after atomic replace/unlink."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


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
        event_store: object,
        transactions_root: "str | Path",
        policy: SwarmCommitPolicy,
        codec: SwarmCommitCodec,
    ) -> None:
        self._state_store = swarm_store
        self._events = event_store
        self._codec = codec
        self._policy = policy
        root = Path(transactions_root)
        self._completed_dir = root / "swarm_completed"
        self._inflight_dir = root / "swarm_inflight"
        self._completed_dir.mkdir(parents=True, exist_ok=True)
        self._inflight_dir.mkdir(parents=True, exist_ok=True)

    @property
    def state_store(self):
        return self._state_store

    async def get_run(self, swarm_run_id: str):
        return await self._state_store.get_run(swarm_run_id)

    async def update_run(self, swarm_run_id: str, *, expected_version: int, status=None, token_usage=None):
        return await self._state_store.update_run(
            swarm_run_id,
            expected_version=expected_version,
            status=status,
            token_usage=token_usage,
        )

    async def recover_incomplete_commits(self) -> None:
        """Recover in-flight journals without deleting unresolved evidence.

        A result-ready journal is safe to forward-complete because the state,
        lifecycle event, and typed result have already crossed their durable
        boundaries. A PREPARED journal cannot prove that the event and
        completion log were written, so recovery marks an active run failed
        when possible but still raises and retains the journal.
        """
        from ..models import SwarmStatus

        failures: list[BaseException] = []
        for path in sorted(self._inflight_dir.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    raise SwarmRecoveryError("swarm journal must contain an object")
                required = {
                    "commit_id", "operation", "swarm_run_id", "request_hash",
                    "state", "command_payload_b64",
                }
                if not required.issubset(raw):
                    raise SwarmRecoveryError("swarm journal is missing required fields")
                commit_id = raw["commit_id"]
                operation = raw["operation"]
                swarm_run_id = raw["swarm_run_id"]
                request_hash = bytes.fromhex(raw["request_hash"])
                if not all(isinstance(value, str) and value for value in (
                    commit_id, operation, swarm_run_id
                )) or len(request_hash) != 32:
                    raise SwarmRecoveryError("swarm journal has invalid identity fields")
                try:
                    base64.b64decode(raw["command_payload_b64"], validate=True)
                except (binascii.Error, TypeError, ValueError) as exc:
                    raise SwarmRecoveryError(
                        "swarm journal has an invalid command payload"
                    ) from exc

                if raw["state"] == "RESULT_READY":
                    result_payload = base64.b64decode(
                        raw.get("result_payload_b64") or "", validate=True
                    )
                    decoded = self._codec.decode_result(operation, result_payload)
                    if not isinstance(decoded, Mapping):
                        raise SwarmRecoveryError("swarm result payload is not a mapping")
                    self._write_completion(
                        commit_id, operation, request_hash, decoded, result_payload
                    )
                    self._clear_inflight(commit_id)
                    continue

                if raw["state"] == "PREPARED":
                    try:
                        current = await self._state_store.get_run(swarm_run_id)
                        if current is not None and current.status in (
                            SwarmStatus.PENDING,
                            SwarmStatus.RUNNING,
                            SwarmStatus.CANCELLING,
                        ):
                            await self._state_store.update_run(
                                swarm_run_id,
                                expected_version=current.version,
                                status=SwarmStatus.FAILED,
                            )
                    except BaseException as exc:
                        failures.append(exc)
                    failures.append(
                        SwarmRecoveryError(
                            f"swarm journal {commit_id!r} stopped before durable completion"
                        )
                    )
                    continue
                raise SwarmRecoveryError(
                    f"unknown swarm journal state {raw['state']!r}"
                )
            except (OSError, TypeError, ValueError, KeyError, SwarmRecoveryError) as exc:
                failures.append(exc)
        if failures:
            raise SwarmRecoveryError(
                f"{len(failures)} swarm journals could not be recovered"
            ) from failures[0]

    def _read_completion(
        self, commit_id: str, request_hash: bytes
    ) -> "Mapping[str, Any] | None":
        path = self._completed_dir / f"{_hash_segment(commit_id)}.json"
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SwarmRecoveryError(
                f"swarm completion log {path.name!r} cannot be parsed"
            ) from exc
        if raw.get("commit_id") != commit_id:
            raise SwarmRecoveryError(
                f"swarm completion log {path.name!r} has an invalid commit id"
            )
        recorded = bytes.fromhex(raw.get("request_hash", ""))
        if recorded != request_hash:
            raise SwarmCommitConflictError(
                f"swarm commit {commit_id!r} replayed with a different request"
            )
        encoded = raw.get("result_payload_b64")
        if encoded:
            try:
                result = self._codec.decode_result(
                    raw["operation"], base64.b64decode(encoded, validate=True)
                )
            except (binascii.Error, KeyError, TypeError, ValueError) as exc:
                raise SwarmRecoveryError(
                    f"swarm completion log {path.name!r} has an invalid result"
                ) from exc
            if not isinstance(result, Mapping):
                raise SwarmRecoveryError(
                    f"swarm completion log {path.name!r} result is not a mapping"
                )
            return result
        return raw.get("result") or {}

    def _write_completion(
        self,
        commit_id: str,
        operation: str,
        request_hash: bytes,
        result: Mapping[str, Any],
        result_payload: bytes | None = None,
    ) -> None:
        path = self._completed_dir / f"{_hash_segment(commit_id)}.json"
        payload = {
            "commit_id": commit_id,
            "operation": operation,
            "request_hash": request_hash.hex(),
            "result": dict(result),
        }
        if result_payload is not None:
            payload["result_payload_b64"] = base64.b64encode(result_payload).decode("ascii")
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        fd = os.open(tmp, os.O_WRONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
        _fsync_parent(path.parent)

    def _write_inflight(
        self,
        commit_id: str,
        operation: str,
        swarm_run_id: str,
        request_hash: bytes,
        command_payload: bytes,
    ) -> None:
        path = self._inflight_dir / f"{_hash_segment(commit_id)}.json"
        payload = {
            "commit_id": commit_id,
            "operation": operation,
            "swarm_run_id": swarm_run_id,
            "request_hash": request_hash.hex(),
            "state": "PREPARED",
            "command_payload_b64": base64.b64encode(command_payload).decode("ascii"),
            "result_payload_b64": None,
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        fd = os.open(tmp, os.O_WRONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
        _fsync_parent(path.parent)

    def _clear_inflight(self, commit_id: str) -> None:
        path = self._inflight_dir / f"{_hash_segment(commit_id)}.json"
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        else:
            _fsync_parent(path.parent)

    def _record_inflight_result(self, commit_id: str, operation: str, result: Mapping[str, Any]) -> bytes:
        payload = self._codec.encode_result(operation, result)
        path = self._inflight_dir / f"{_hash_segment(commit_id)}.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["state"] = "RESULT_READY"
        raw["result_payload_b64"] = base64.b64encode(payload).decode("ascii")
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")
        fd = os.open(tmp, os.O_WRONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
        _fsync_parent(path.parent)
        return payload

    def _check_replay(
        self, commit_id: str, operation: str, request_payload: Mapping[str, Any]
    ) -> "Mapping[str, Any] | None":
        return self._read_completion(commit_id, self._codec.request_hash(operation, request_payload))

    def _record(
        self,
        commit_id: str,
        operation: str,
        swarm_run_id: str,
        request_payload: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> None:
        request_hash = self._codec.request_hash(operation, request_payload)
        self._write_completion(commit_id, operation, request_hash, result)
        self._clear_inflight(commit_id)

    async def _run_commit(
        self,
        operation: str,
        command: Any,
        run_business: Any,
        *,
        establish_run_owner: bool = False,
    ) -> "SwarmCommitResult":
        """Wrap a business-write closure with replay detection + inflight
        journaling + completion-log recording. ``run_business`` is a sync or
        async callable that performs the actual swarm_store op and returns
        the result dict to record.

        Fencing order: replay first (an already-completed commit returns even
        after the token rotates), THEN -- for a fresh commit on an existing
        run -- the run-level owner fence (assert_execution_fence + policy
        validate) BEFORE the inflight journal or any business write.
        ``establish_run_owner=True`` (start) skips the fence (the run is being
        created) and instead stamps the supplied token as the persisted
        execution_token."""
        commit_id = str(command.commit_id)
        swarm_run_id = command.swarm_run_id
        request_hash = self._codec.request_hash(operation, command)
        replay = self._read_completion(commit_id, request_hash)
        if replay is not None:
            return replay
        if not establish_run_owner:
            current = await self._state_store.assert_execution_fence(
                swarm_run_id, expected_token=command.fence.token
            )
            self._policy.validate(
                supplied=command.fence, stored_token=current.execution_token
            )
        # Journal the intent BEFORE the business write so a crash mid-write
        # is reconcilable (recovery marks the swarm_run FAILED).
        self._write_inflight(
            commit_id,
            operation,
            swarm_run_id,
            request_hash,
            self._codec.encode_request(operation, command),
        )
        result = await run_business()
        payload = getattr(command, "payload", None)
        event_context = getattr(payload, "event_context", None)
        lifecycle_event = next(
            (
                getattr(payload, name, None)
                for name in (
                    "started_event", "step_event", "completed_event",
                    "failed_event", "cancelled_event",
                )
                if getattr(payload, name, None) is not None
            ),
            None,
        )
        if event_context is not None and lifecycle_event is not None:
            from ...events.context import append_event_once
            await append_event_once(
                self._events,
                event_context,
                lifecycle_event,
                commit_id=commit_id,
                metadata={"commit_id": commit_id},
            )
        result_payload = self._record_inflight_result(commit_id, operation, result)
        self._write_completion(
            commit_id,
            operation,
            request_hash,
            result,
            result_payload,
        )
        self._clear_inflight(commit_id)
        return result

    async def start(self, command: "StartSwarmCommand") -> "SwarmCommitResult":
        async def _business():
            from dataclasses import replace as _replace
            from ..models import SwarmRun

            # start ESTABLISHES the run-level owner: stamp the supplied
            # fence token as the persisted execution_token.
            run_to_create = _replace(
                command.payload.run, execution_token=command.fence.token
            )
            created = await self._state_store.create_run(run_to_create)
            return {"swarm_run_id": created.id, "version": created.version}

        return await self._run_commit(
            "start", command, _business, establish_run_owner=True
        )

    async def start_step(self, command: "StartSwarmStepCommand") -> "SwarmCommitResult":
        async def _business():
            from ..models import SwarmStep

            created = await self._state_store.create_task(command.payload.step)
            return {"task_id": created.id, "version": created.version}

        return await self._run_commit(
            "start_step", command, _business
        )

    async def complete_step(self, command: "CompleteSwarmStepCommand") -> "SwarmCommitResult":
        async def _business():
            from ...run.models import RunResult

            task_id = command.payload.task_id
            result = command.payload.result
            updated = await self._state_store.complete_task(
                task_id,
                result,
                expected_version=command.expected_version,
                active_run_id=command.payload.active_run_id,
            )
            return {"task_id": updated.id, "version": updated.version}

        return await self._run_commit(
            "complete_step", command, _business
        )

    async def fail_step(self, command: "FailSwarmStepCommand") -> "SwarmCommitResult":
        async def _business():
            from ...run.models import RunErrorInfo

            task_id = command.payload.task_id
            error = command.payload.error
            updated = await self._state_store.fail_task(
                task_id,
                error,
                expected_version=command.expected_version,
                active_run_id=command.payload.active_run_id,
            )
            return {"task_id": updated.id, "version": updated.version}

        return await self._run_commit(
            "fail_step", command, _business
        )

    async def complete(self, command: "CompleteSwarmCommand") -> "SwarmCommitResult":
        async def _business():
            from ..models import SwarmStatus

            updated = await self._state_store.update_run(
                command.swarm_run_id,
                expected_version=command.expected_version,
                status=SwarmStatus.SUCCEEDED,
            )
            return {"swarm_run_id": updated.id, "version": updated.version}

        return await self._run_commit(
            "complete", command, _business
        )

    async def fail(self, command: "FailSwarmCommand") -> "SwarmCommitResult":
        async def _business():
            from ..models import SwarmStatus

            updated = await self._state_store.update_run(
                command.swarm_run_id,
                expected_version=command.expected_version,
                status=SwarmStatus.FAILED,
            )
            return {"swarm_run_id": updated.id, "version": updated.version}

        return await self._run_commit(
            "fail", command, _business
        )

    async def cancel(self, command: "CancelSwarmCommand") -> "SwarmCommitResult":
        async def _business():
            from ..models import SwarmStatus

            updated = await self._state_store.update_run(
                command.swarm_run_id,
                expected_version=command.expected_version,
                status=SwarmStatus.CANCELLED,
            )
            return {"swarm_run_id": updated.id, "version": updated.version}

        return await self._run_commit(
            "cancel", command, _business
        )


__all__: "list[str]" = ["FilesystemSwarmCommitCoordinator"]


class SwarmRecoveryError(RuntimeError):
    pass
