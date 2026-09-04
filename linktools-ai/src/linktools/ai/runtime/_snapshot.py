#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical Runtime run snapshot digest."""

from dataclasses import dataclass

from ..core import JsonValue, canonical_sha256


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    snapshot_id: str
    execution_id: str
    binding_digest: str
    trace_digest: str
    result_digest: "str | None"
    digest: str

    def verify(self) -> bool:
        return self.digest == snapshot_digest(
            {
                "snapshot_id": self.snapshot_id,
                "execution_id": self.execution_id,
                "binding_digest": self.binding_digest,
                "trace_digest": self.trace_digest,
                "result_digest": self.result_digest,
            }
        )


def snapshot_digest(value: JsonValue) -> str:
    return canonical_sha256(value)


__all__ = ["RunSnapshot", "snapshot_digest"]
