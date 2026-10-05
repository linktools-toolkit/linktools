#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scoped references to immutable Task invocation and graph captures."""

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TaskInvocationInputRef:
    namespace: str
    tenant_id: str
    capture_id: str
    digest: str
    source_execution_id: str

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value for value in (
            self.namespace, self.tenant_id, self.capture_id, self.source_execution_id,
        )) or re.fullmatch(r"[0-9a-f]{64}", self.digest) is None:
            raise ValueError("task input capture reference is invalid")


@dataclass(frozen=True, slots=True)
class TaskGraphCaptureRef:
    namespace: str
    tenant_id: str
    capture_id: str
    digest: str
    source_graph_id: str

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value for value in (
            self.namespace, self.tenant_id, self.capture_id, self.source_graph_id,
        )) or re.fullmatch(r"[0-9a-f]{64}", self.digest) is None:
            raise ValueError("task graph capture reference is invalid")


@dataclass(frozen=True, slots=True)
class TaskGraphTemplateRef:
    namespace: str
    tenant_id: str
    capture_id: str
    digest: str

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value for value in (
            self.namespace, self.tenant_id, self.capture_id,
        )) or re.fullmatch(r"[0-9a-f]{64}", self.digest) is None:
            raise ValueError("task graph template reference is invalid")


__all__ = ["TaskGraphTemplateRef", "TaskInvocationInputRef", "TaskGraphCaptureRef"]
