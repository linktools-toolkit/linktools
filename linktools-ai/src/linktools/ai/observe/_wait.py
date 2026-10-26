#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable SDK result of an authoritative wait and optional observation."""

from dataclasses import dataclass
from typing import Generic, TypeVar

from ..errors import ObservationError

ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class WaitResult(Generic[ResultT]):
    result: ResultT
    cursor: str | None
    observation_error: ObservationError | None = None


__all__ = ["WaitResult"]
