#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional fixed-edge reads for immutable archive records."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntFlag
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ._record_key import validate_record_key

if TYPE_CHECKING:
    from .state import StoredRecord


class ArchiveNodeSlot(IntFlag):
    SELF = 1
    LEFT = 2
    RIGHT = 4
    NEXT = 8


@dataclass(frozen=True, slots=True)
class ArchiveNodeRead:
    key: bytes = field(repr=False)
    metadata_slots: ArchiveNodeSlot = ArchiveNodeSlot.SELF
    content_slots: ArchiveNodeSlot = ArchiveNodeSlot(0)

    def __post_init__(self) -> None:
        validate_record_key(self.key)
        for value in (self.metadata_slots, self.content_slots):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 15:
                raise ValueError("archive read slots must contain only SELF, LEFT, RIGHT, NEXT")
        content = ArchiveNodeSlot(self.content_slots)
        metadata = ArchiveNodeSlot(self.metadata_slots) | content | ArchiveNodeSlot.SELF
        object.__setattr__(self, "metadata_slots", metadata)
        object.__setattr__(self, "content_slots", content)


@runtime_checkable
class ArchiveFrontierReader(Protocol):
    async def read_archive_frontiers(
        self, *, scope_key: bytes, requests: Sequence[ArchiveNodeRead],
        content_keys: Sequence[bytes] = (),
    ) -> "Mapping[bytes, StoredRecord] | None":
        """Return selected scoped records, or None only when unsupported."""
        ...


__all__ = ["ArchiveNodeSlot", "ArchiveNodeRead", "ArchiveFrontierReader"]
