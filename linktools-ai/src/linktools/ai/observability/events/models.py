#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared event value types with no registry or codec dependencies."""

from enum import Enum


class EventCriticality(str, Enum):
    STATE_CRITICAL = "state_critical"
    SECURITY_CRITICAL = "security_critical"
    OBSERVABILITY = "observability"


__all__ = ["EventCriticality"]
