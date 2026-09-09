#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit tool-effect outcome markers."""

from pydantic_ai.exceptions import ToolFailed


class ToolEffectNotAppliedError(ToolFailed):
    """Signal that a tool call failed before producing an external effect."""


__all__ = ["ToolEffectNotAppliedError"]
