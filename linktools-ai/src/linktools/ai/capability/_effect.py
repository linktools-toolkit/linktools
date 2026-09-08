#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit tool-effect outcome markers."""


class ToolEffectNotAppliedError(Exception):
    """Signal that a tool call failed before producing an external effect."""


__all__ = ["ToolEffectNotAppliedError"]
