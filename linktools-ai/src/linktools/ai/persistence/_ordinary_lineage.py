#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable canonical turn ranges shared by ordinary session forks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..core import validate_idempotency_key, validate_session_id
from ..errors import AIError, ErrorCode

if TYPE_CHECKING:
    from sqlalchemy import ColumnCollection, ColumnElement

# A single SQL query can seek each non-overlapping range in the existing
# session/turn index. Bound fan-out without copying inherited rows.
_MAX_SEGMENTS = 32


@dataclass(frozen=True, slots=True)
class SessionTurnSegment:
    session_id: str
    after: int
    through: int


def decode_session_lineage(raw: object) -> tuple[SessionTurnSegment, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, dict) or set(raw) != {"source", "request_id", "source_revision", "segments"}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    segments = raw["segments"]
    revision = raw["source_revision"]
    if type(revision) is not int or revision < 0:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if not isinstance(segments, list) or len(segments) > _MAX_SEGMENTS:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        validate_session_id(raw["source"])
        validate_idempotency_key(raw["request_id"])
        result: list[SessionTurnSegment] = []
        positions = 0
        owners: set[str] = set()
        for value in segments:
            session_id, after, through = value
            validate_session_id(session_id)
            if (session_id in owners or type(after) is not int or type(through) is not int
                    or positions != after or not after < through < 2**63):
                raise ValueError("invalid ordinary session lineage")
            result.append(SessionTurnSegment(session_id, after, through))
            positions = through
            owners.add(session_id)
    except (TypeError, ValueError, AIError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    return tuple(result)


def fork_segments(
    source_id: str, lineage: tuple[SessionTurnSegment, ...], through: int,
) -> tuple[SessionTurnSegment, ...]:
    after = lineage[-1].through if lineage else 0
    if through < after:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if through == after:
        return lineage
    if len(lineage) >= _MAX_SEGMENTS:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, "Too many inherited session segments")
    return (*lineage, SessionTurnSegment(source_id, after, through))


def lineage_predicate(
    columns: ColumnCollection, session_key: bytes, lineage: tuple[SessionTurnSegment, ...],
    *, after: int, through: int,
) -> ColumnElement[bool]:
    from sqlalchemy import and_, or_

    own_after = lineage[-1].through if lineage else 0
    return or_(
        and_(columns.session_key == session_key, columns.turn_no > max(own_after, after),
             columns.turn_no <= through),
        *(and_(columns.session_key == segment.session_id.encode("utf-8"),
               columns.turn_no > max(segment.after, after), columns.turn_no <= min(segment.through, through))
          for segment in lineage if segment.through > after and segment.after < through),
    )
