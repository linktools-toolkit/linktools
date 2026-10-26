#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared disposable Session display cache with current membership validation."""

import json
import math
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import datetime

from ..core import (
    CursorSigner,
    ExecutionStatus,
    Page,
    Principal,
    ResourceKind,
    ResourceRef,
    SessionStatus,
    canonical_sha256,
    normalize_json_value,
)
from ..errors import AIError, ErrorCode
from ..session import SessionDetail, SessionManager, SessionTurn, SessionTurnItem, SessionView
from ..storage import CacheBackend
from ..inputs import stored_user_input_view
from ._session_timeline import SessionTimelineCapture, SessionTimelineTranscriptStore
from linktools.ai.persistence.runs import ConversationPersistence
from linktools.ai.persistence.codec import decode_domain, encode_domain
from linktools.ai.persistence.contracts import ConversationRepositories, ExecutionRepository
from linktools.ai.execution import ExecutionRecord
from linktools.ai.session import SessionRecord, SessionTurnCommit
from linktools.ai.persistence.repositories import ConversationHistoryRepositoryImpl, SessionRepositoryImpl
from linktools.ai.persistence.repositories import ExecutionRepositoryImpl
from linktools.ai.persistence.runs import ModelInteractionStateStepArchive
from linktools.ai.persistence.runs import StateStepArchive


class SessionDisplayCache:
    """Cache presentation only; never authorize or reconcile durable state."""

    def __init__(
        self, conversation: ConversationRepositories, executions: ExecutionRepository,
        transcript_store: SessionTimelineTranscriptStore | None, cursor_signer: CursorSigner,
        cache: CacheBackend | None, *, now: Callable[[], float],
    ) -> None:
        self._conversation = conversation
        self._executions = executions
        self._transcript_store = transcript_store
        self._cursor_signer = cursor_signer
        self._cache = cache
        self._now = now

    def supported(self) -> bool:
        sessions = self._conversation.sessions
        histories = self._conversation.histories
        executions = self._executions
        if (self._cache is None
                or type(sessions) is not SessionRepositoryImpl
                or type(histories) is not ConversationHistoryRepositoryImpl
                or type(executions) is not ExecutionRepositoryImpl
                or type(self._transcript_store) is not ConversationPersistence):
            return False
        archive = self._transcript_store.archive
        return (
            type(archive) in (StateStepArchive, ModelInteractionStateStepArchive)
            and archive.state_store is sessions.state_store
            and histories.state_store is sessions.state_store
            and sessions.state_store.storage_group is executions.state_store.storage_group
        )

    @staticmethod
    def _display_session_identity(record: SessionRecord) -> tuple[object, ...]:
        return (record.session_id, record.owner_principal_id, record.agent_id,
                record.history_id, record.created_at)

    @staticmethod
    def _display_execution_identity(record: ExecutionRecord) -> tuple[object, ...]:
        return (record.execution_id, record.session_id, record.parent_execution_id,
                record.created_at, record.principal_id, record.principal_kind,
                record.binding_digest)


    @staticmethod
    def _display_commit_identity(record: SessionTurnCommit) -> tuple[int, str, str, datetime]:
        return record.turn_seq, record.session_id, record.execution_id, record.created_at

    @staticmethod
    def _decode_display(value: object) -> SessionDetail:
        # These public view dataclasses intentionally do not enforce wire types.
        # Validate the disposable representation without registering durable types.
        def text(value: object, *, optional: bool = False) -> None:
            if not isinstance(value, str) and not (optional and value is None):
                raise TypeError("invalid display text")

        session, timeline = value["session"], value["timeline"]
        for name in ("session_id", "agent_id"):
            text(session[name])
        for name in ("cwd", "active_execution_id"):
            text(session[name], optional=True)
        if (type(session["revision"]) is not int or session["revision"] < 0
                or session["history_quality"] not in {"complete", "conservative", "unavailable"}
                or not isinstance(session["metadata"], dict)):
            raise ValueError("invalid session display")
        view = SessionView(**{
            **session, "status": SessionStatus(session["status"]),
            "metadata": normalize_json_value(session["metadata"]),
        })
        text(timeline["next_cursor"], optional=True)
        turns = []
        for turn in timeline["items"]:
            text(turn["execution_id"])
            text(turn["error_code"], optional=True)
            if (not all(isinstance(turn[key], datetime) for key in ("created_at", "updated_at"))
                    or type(turn["conversation_committed"]) is not bool
                    or not isinstance(turn["safe_error_details"], dict)):
                raise ValueError("invalid turn display")
            items = []
            for item in turn["items"]:
                if type(item["ordinal"]) is not int:
                    raise TypeError("invalid item ordinal")
                text(item["item_kind"])
                text(item["tool_name"], optional=True)
                text(item["tool_call_id"], optional=True)
                items.append(SessionTurnItem(**{**item, "content": normalize_json_value(item["content"])}))
            turns.append(SessionTurn(**{
                **turn, "status": ExecutionStatus(turn["status"]), "items": tuple(items),
                "user_input": normalize_json_value(turn["user_input"]),
                "output": normalize_json_value(turn["output"]),
                "safe_error_details": normalize_json_value(turn["safe_error_details"]),
            }))
        return SessionDetail(view, Page(tuple(turns), timeline["next_cursor"]))

    async def read(
        self, key: str, header: ResourceRef, principal: Principal,
        *, cursor: str | None, max_age: float, pending: ExecutionRecord | None = None,
    ) -> SessionDetail | None:
        try:
            raw = await self._cache.get(key)
            if raw is None:
                return None
            envelope = json.loads(raw)
            if canonical_sha256(envelope["payload"]) != envelope["digest"]:
                return None
            entry = decode_domain(envelope["payload"], dict[str, object])
            started, expires = entry["started"], entry["expires"]
            if (entry["key"] != key or not isinstance(started, (float, int))
                    or not isinstance(expires, (float, int))
                    or not math.isfinite(started) or not math.isfinite(expires)
                    or not started <= self._now() < min(expires, started + max_age)):
                return None
            pending_identity = None if pending is None else self._display_execution_identity(pending)
            if entry.get("pending") != pending_identity:
                return None
            detail = self._decode_display(entry["detail"])
            if detail.session.session_id != header.id:
                return None
            # The exact input token was validated when this entry was captured;
            # decode it again for expiration/key rotation, without extending it.
            for token in (cursor, detail.timeline.next_cursor):
                if token is not None:
                    self._cursor_signer.decode(token)
            identities = entry["executions"]
            commits = entry["commits"]
            if (not isinstance(identities, dict) or not all(isinstance(key, str) for key in identities)
                    or not isinstance(commits, dict) or not all(isinstance(key, str) for key in commits)
                    or set(identities) & set(commits)):
                return None
            for execution_id, identity in commits.items():
                sequence, source_session_id, committed_execution_id, created_at = identity
                if (type(sequence) is not int or sequence < 1
                        or source_session_id != header.id or committed_execution_id != execution_id
                        or not isinstance(created_at, datetime)):
                    return None
            if len({identity[0] for identity in commits.values()}) != len(commits):
                return None
            if (detail.session.active_execution_id is not None and detail.session.active_execution_id not in identities
                    or not all(
                        turn.execution_id in (commits if turn.conversation_committed else identities)
                        for turn in detail.timeline.items
                    )
                    or detail.session.session_id != entry["root"][0]
                    or detail.session.agent_id != entry["root"][2]):
                return None
            required_end = entry["message_end"]
            if not isinstance(required_end, int) or required_end < 0:
                return None
        except Exception:  # noqa: BLE001 -- a disposable cache never replaces origin errors
            return None

        sessions = self._conversation.sessions
        histories = self._conversation.histories
        async def validate() -> bool:
            if pending is not None:
                # Another writer may commit without invalidating this reader's
                # negative cache while the old coordination lease still lives.
                canonical = await self._executions.get_many_fresh(
                    (pending.execution_id,), tenant_id=principal.tenant_id,
                )
                if canonical:
                    return False
            record = await sessions.get(header.id, tenant_id=principal.tenant_id)
            if record is None:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            if (header.kind is not ResourceKind.SESSION or record.session_id != header.id
                    or record.owner_principal_id != header.owner_principal_id):
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            if (record.timeline_parent_session_id is not None
                    or self._display_session_identity(record) != entry["root"]
                    or record.revision > detail.session.revision):
                return False
            ids = tuple(identities)
            current_revision = record.revision == detail.session.revision
            if current_revision and record.active_execution_id is not None and record.active_execution_id not in identities:
                ids += (record.active_execution_id,)
            records = await self._executions.get_many(ids, tenant_id=principal.tenant_id) if ids else {}
            if len(records) != len(ids):
                # A pre-admission negative cache entry can outlive the source
                # capture. Only authoritative absence is an integrity failure.
                records = await self._executions.get_many_fresh(ids, tenant_id=principal.tenant_id)
            if len(records) != len(ids):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if current_revision:
                active = SessionManager.validate_admission(
                    record, records.get(record.active_execution_id) if record.active_execution_id else None,
                )
                if (record.active_execution_id != detail.session.active_execution_id
                        or record.active_execution_id is not None and active is None):
                    return False
            # A source-captured display can be newer than another reader's
            # bounded session cache. Do not replace that newer whole snapshot
            # merely because the same resource's older revision is still hot.
            if any(value.session_id != record.session_id or value.parent_execution_id is not None
                   for value in records.values()):
                return False
            if any(self._display_execution_identity(records[key]) != identity
                   for key, identity in identities.items()):
                return False
            if commits:
                sequences = tuple(identity[0] for identity in commits.values())
                current_commits = await sessions.list_timeline_commits(
                    record.session_id, tenant_id=principal.tenant_id,
                    start_turn_seq=min(sequences), end_turn_seq=max(sequences) + 1,
                )
                retained = {value.execution_id: self._display_commit_identity(value) for value in current_commits}
                if any(retained.get(key) != identity for key, identity in commits.items()):
                    return False
            if required_end == 0:
                return True
            if record.history_id is None:
                return False
            value = await histories.get_with_head(
                record.history_id, tenant_id=principal.tenant_id,
            )
            if (value is None or value[0].session_id != record.session_id
                    or value[0].parent_history_id is not None or value[1].message_count < required_end):
                # A pending-page head may predate the committed display. Confirm
                # source membership and replace that older ordinary cache batch.
                value = await histories.get_with_head_fresh(record.history_id, tenant_id=principal.tenant_id)
            if value is None:
                raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
            history, head = value
            return (history.session_id == record.session_id and history.parent_history_id is None
                    and head.message_count >= required_end)

        # Reevaluate policy above on every request; only immutable resource
        # identity/membership and display state use the existing bounded cache.
        # A mismatch rebuilds the page in the authoritative capture below.
        valid = await validate()
        if valid and self._now() < min(expires, started + max_age):
            if pending is not None:
                # Only the canonical page was cached. Progress and status always
                # come from this request's authorized pending publication.
                turn = SessionTurn(
                    pending.execution_id, pending.status, pending.created_at, pending.updated_at,
                    stored_user_input_view(pending.stored_user_input), False, (),
                    pending.error_code, pending.safe_error_details,
                )
                return SessionDetail(
                    replace(detail.session, active_execution_id=pending.execution_id),
                    Page((*detail.timeline.items, turn), detail.timeline.next_cursor),
                )
            return detail
        return None

    async def write(
        self, key: str, state: SessionTimelineCapture, detail: SessionDetail,
        started: float, max_age: float,
    ) -> None:
        if state.root.history_id is None or self._now() >= started + max_age:
            return
        pending = state.pending_execution
        if pending is not None:
            # Root pages reserve exactly one slot for the live pending turn.
            # Its mutable content is never included in the cached page.
            if not detail.timeline.items or detail.timeline.items[-1].execution_id != pending.execution_id:
                return
            detail = SessionDetail(
                replace(detail.session, active_execution_id=None),
                Page(detail.timeline.items[:-1], detail.timeline.next_cursor),
            )
        try:
            entry = {
                "key": key, "started": started, "expires": started + max_age,
                "root": self._display_session_identity(state.root),
                "executions": {key: self._display_execution_identity(value)
                               for key, value in state.executions.items()
                               if pending is None or key != pending.execution_id},
                "commits": {value.execution_id: self._display_commit_identity(value)
                            for value in state.commits.values()},
                "pending": None if pending is None else self._display_execution_identity(pending),
                "message_end": max((value.end_message_index for value in state.commits.values()), default=0),
                "detail": asdict(detail),
            }
            payload = encode_domain(entry)
            envelope = {"payload": payload, "digest": canonical_sha256(payload)}
            await self._cache.put(key, json.dumps(envelope, separators=(",", ":")).encode())
        except Exception:  # noqa: BLE001 -- best-effort serialization/publication only
            return




__all__ = ["SessionDisplayCache"]
