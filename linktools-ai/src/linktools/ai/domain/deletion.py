#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Auditable cross-domain deletion state."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class DeletionStatus(StrEnum):
    REQUESTED = "REQUESTED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class DeletionJob(BaseModel):
    model_config = ConfigDict(frozen=True)

    deletion_id: str
    scopes: "tuple[str, ...]"
    status: DeletionStatus = DeletionStatus.REQUESTED
    evidence: "tuple[str, ...]" = ()

    def advance(self, target: DeletionStatus, evidence: "tuple[str, ...]" = ()) -> "DeletionJob":
        allowed = {
            DeletionStatus.REQUESTED: {DeletionStatus.RUNNING, DeletionStatus.FAILED},
            DeletionStatus.RUNNING: {DeletionStatus.SUCCEEDED, DeletionStatus.FAILED},
            DeletionStatus.SUCCEEDED: set(),
            DeletionStatus.FAILED: set(),
        }
        if target not in allowed[self.status]:
            raise ValueError(f"invalid deletion transition: {self.status} -> {target}")
        return self.model_copy(update={"status": target, "evidence": self.evidence + evidence})


__all__ = ["DeletionJob", "DeletionStatus"]
