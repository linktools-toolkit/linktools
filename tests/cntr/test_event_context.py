#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EventContext typed dataclass (refactor spec Phase 8).

Plain (non-frozen, non-slots) dataclass: defaults match the previous __init__,
dynamic attribute assignment still works for legacy hooks, and ``metadata`` is an
opt-in extension field.
"""
from dataclasses import is_dataclass

from linktools.cntr.context import EventContext


def test_is_dataclass_with_legacy_defaults():
    ctx = EventContext()
    assert is_dataclass(EventContext)
    assert ctx.commands is None
    assert ctx.containers is None
    assert ctx.target_containers is None
    # is_full_containers default kept as True (every caller sets it explicitly).
    assert ctx.is_full_containers is True


def test_metadata_defaults_to_independent_dict():
    a = EventContext()
    b = EventContext()
    a.metadata["k"] = 1
    assert b.metadata == {}  # default_factory -> per-instance, not shared


def test_dynamic_attribute_assignment_still_works():
    # Legacy/third-party hooks may set arbitrary attributes (refactor spec §12.2).
    ctx = EventContext()
    ctx.custom_field = "value"
    assert ctx.custom_field == "value"
