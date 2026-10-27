#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Logical persistence ownership shared by domains and storage composition."""
from enum import Enum


class PersistenceDomain(str, Enum):
    __str__ = str.__str__
    __format__ = str.__format__
    CONVERSATION = "conversation"
    EXECUTION = "execution"
    MEMORY = "memory"
    ARTIFACT = "artifact"
    TASK = "task"
    EVALUATION = "evaluation"
    RECOVERY = "recovery"
    TRACE = "trace"
    CHECKPOINT = "checkpoint"


__all__ = ["PersistenceDomain"]
