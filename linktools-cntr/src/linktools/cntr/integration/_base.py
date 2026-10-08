#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared declaration metadata."""
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from typing import Optional


class Integration:
    """A consumer-specific declaration with an optional producer-local ID."""

    consumer: str = ""
    local_id: "Optional[str]" = None
    requires_local_id: bool = False


Integrations = Iterable[Integration]
