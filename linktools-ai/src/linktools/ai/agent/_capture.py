#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable references to Runtime-owned accepted Agent inputs."""

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgentInputCaptureRef:
    namespace: str
    tenant_id: str
    capture_id: str
    digest: str
    source_execution_id: str | None

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value for value in (
            self.namespace, self.tenant_id, self.capture_id,
        )) or re.fullmatch(r"[0-9a-f]{64}", self.digest) is None:
            raise ValueError("agent input capture reference is invalid")
        if self.source_execution_id is not None and (not isinstance(self.source_execution_id, str) or not self.source_execution_id):
            raise ValueError("capture source execution is invalid")


__all__ = ["AgentInputCaptureRef"]
