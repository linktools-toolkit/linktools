#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared declaration metadata."""
from typing import Iterable


class Integration:
    """A consumer-specific declaration."""

    consumer: str = ""


Integrations = Iterable[Integration]
