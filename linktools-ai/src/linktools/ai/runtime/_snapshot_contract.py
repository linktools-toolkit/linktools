#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transport-neutral execution snapshot values."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    snapshot_id: str
    execution_id: str
    binding_digest: str


__all__ = ["RunSnapshot"]
