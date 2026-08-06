#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pure deletion-retention policy."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DeletionPolicy:
    retain_audit_evidence: bool = True

    def evaluate(self, execution_id: "str | None", conversation_id: "str | None") -> "tuple[str, ...]":
        scopes = tuple(value for value in (execution_id, conversation_id) if value is not None)
        if not scopes:
            raise ValueError("a deletion scope is required")
        return scopes


__all__ = ["DeletionPolicy"]
