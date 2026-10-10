#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared declaration metadata."""
from typing import Iterable, Mapping, Union


class Integration:
    """A consumer-specific declaration."""

    consumer: str = ""


Integrations = Union[Mapping[str, Integration], Iterable[Union[Mapping[str, Integration], Integration]]]
