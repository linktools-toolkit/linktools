#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime domain names shared by service contracts and state implementations."""

from enum import StrEnum


class RuntimeDomain(StrEnum):
    CONVERSATION = "conversation"
    EXECUTION = "execution"
    MEMORY = "memory"
    ARTIFACT = "artifact"
    TASK = "task"
    EVALUATION = "evaluation"
    RECOVERY = "recovery"


__all__ = ["RuntimeDomain"]
