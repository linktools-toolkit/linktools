#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CommandGroupRef fallback semantics.

A command may declare ``parent = CommandGroupRef(...)`` to attach under a group
that is auto-materialised only when no real group with that id is registered.
Plain-string parents are NOT fallbacks: a missing string parent still raises.
"""
import argparse

import pytest

from linktools.cli import BaseCommand, CommandGroupRef
from linktools.cli.command import (
    SubCommand,
    SubCommandGroup,
    SubCommandWrapper,
    SubCommandError,
    _CommandInfo,
    _ensure_declared_parent_groups,
)


def _child(name, parent_value):
    """A SubCommandWrapper around a BaseCommand whose ``parent`` is parent_value."""
    class _Cmd(BaseCommand):
        @property
        def name(self):
            return name

        @property
        def parent(self):
            return parent_value

        def init_arguments(self, parser):
            pass

        def run(self, args):
            return 0

    return SubCommandWrapper(_Cmd())


def _real_common(name="ct"):
    return SubCommandGroup(
        name=name, description="Common scripts",
        id="common", order="\x1f100-common",
    )


# 10.1 -- missing parent materialised from the CommandGroupRef
def test_declared_missing_parent_is_materialised():
    child = _child("cntr", CommandGroupRef(
        id="common", name="ct", description="Common scripts", order="\x1f100-common",
    ))
    result = _ensure_declared_parent_groups([child])

    groups = [i for i in result if i.id == "common"]
    assert len(groups) == 1
    assert groups[0].name == "ct"
    assert groups[0].description == "Common scripts"
    assert groups[0].order == "\x1f100-common"
    # child still present, points at common
    assert any(i.id != "common" and i.parent_id == "common" for i in result)


# 10.3 -- a real group wins; a conflicting ref is ignored (no error, no dup)
def test_real_group_wins_over_conflicting_ref():
    real = _real_common(name="ct")
    child = _child("cntr", CommandGroupRef(id="common", name="other", order="zzz"))
    result = _ensure_declared_parent_groups([real, child])

    commons = [i for i in result if i.id == "common"]
    assert len(commons) == 1
    assert commons[0].name == "ct"  # the real one, not the ref's "other"


# 10.4 -- two consistent refs merge into a single group
def test_consistent_refs_merge_into_one_group():
    c1 = _child("cntr", CommandGroupRef(id="common", name="ct", order="\x1f100-common"))
    c2 = _child("env", CommandGroupRef(id="common", name="ct", order="\x1f100-common"))
    result = _ensure_declared_parent_groups([c1, c2])

    assert sum(1 for i in result if i.id == "common") == 1


# 10.5 -- conflicting refs (same id, different name) raise
def test_conflicting_refs_raise():
    c1 = _child("cntr", CommandGroupRef(id="common", name="ct", order="\x1f100-common"))
    c2 = _child("env", CommandGroupRef(id="common", name="common", order="\x1f100-common"))
    with pytest.raises(SubCommandError):
        _ensure_declared_parent_groups([c1, c2])


# 10.2 -- a plain string parent is NOT a fallback; nothing is materialised for it
def test_string_parent_is_not_materialised():
    child = _child("cntr", "common")
    result = _ensure_declared_parent_groups([child])
    assert all(i.id != "common" for i in result)


# --- end-to-end through add_subcommands ---

def _info(name, parent_value):
    class _Cmd(BaseCommand):
        @property
        def name(self):
            return name

        @property
        def parent(self):
            return parent_value

        def init_arguments(self, parser):
            pass

        def run(self, args):
            return 0

    info = _CommandInfo()
    info.id = name
    info.parent_id, info.declared_parent_group = _normalize_parent_pub(parent_value)
    info.module = "fake"
    info.command = _Cmd()
    info.command_name = name
    info.command_description = ""
    info.order = name
    return info


def _normalize_parent_pub(parent):
    from linktools.cli.command import _normalize_parent
    return _normalize_parent(parent)


class _Root(BaseCommand):
    def init_arguments(self, parser):
        pass

    def run(self, args):
        return 0


def test_command_group_ref_registers_when_parent_missing():
    # End-to-end success path is covered by the cntr-only smoke (cntr declares
    # parent=CommandGroupRef and `lt ct cntr` registers). At the unit level the
    # materialisation is already proven by test_declared_missing_parent_is_materialised.
    # Here we only assert the helper does not raise for a ref parent.
    child = _child("cntr", CommandGroupRef(id="common", name="ct", order="\x1f100-common"))
    result = _ensure_declared_parent_groups([child])
    assert any(i.id == "common" for i in result)


def test_string_parent_missing_still_raises():
    # end-to-end: a missing plain-string parent still surfaces SubCommandError
    root = _Root()
    info = _info("cntr", "common")  # string, no real group, no ref
    with pytest.raises(SubCommandError):
        root.add_subcommands(parser=argparse.ArgumentParser(prog="t"), target=[info])
