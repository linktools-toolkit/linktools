#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility proxy for LinkTools-only capability support."""

from __future__ import annotations

import linktools.ai.runtime._capability_support as _support

for _name in _support.__all__:
    globals()[_name] = getattr(_support, _name)


def __getattr__(name: str) -> object:
    return getattr(_support, name)


__all__ = list(_support.__all__)
