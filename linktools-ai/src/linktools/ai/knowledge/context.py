#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KnowledgeContext + format_memory: render retrieved docs / memories into a prompt section."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .document import Document

if TYPE_CHECKING:
    from ..memory.models import MemoryRecord


@dataclass(frozen=True, slots=True)
class KnowledgeContext:
    documents: "tuple[Document, ...]"

    def format(self) -> str:
        if not self.documents:
            return ""
        lines = ["## Knowledge"]
        for doc in self.documents:
            lines.append(f"- {doc.content}")
        return "\n".join(lines)


def format_memory(records: "tuple[MemoryRecord, ...]") -> str:
    if not records:
        return ""
    lines = ["## Memory"]
    for record in records:
        lines.append(f"- {record.content}")
    return "\n".join(lines)
