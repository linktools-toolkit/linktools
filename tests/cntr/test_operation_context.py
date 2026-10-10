#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OperationContext shared state and public import contract.

Plain (non-frozen, non-slots) dataclass so dynamic attribute assignment still
works for hooks that set arbitrary attributes; ``metadata`` is an opt-in
extension field.
"""
from dataclasses import is_dataclass

from linktools.cntr import OperationContext
from linktools.cntr.context import OperationContext as ContextType


def test_is_dataclass_with_operation_defaults():
    ctx = OperationContext()
    assert is_dataclass(OperationContext)
    assert OperationContext is ContextType
    assert ctx.actions is None
    assert ctx.project_containers is None
    assert ctx.target_containers is None
    # is_full_project default is True; every caller sets it explicitly.
    assert ctx.is_full_project is True
    assert ctx.target_services is None


def test_metadata_defaults_to_independent_dict():
    a = OperationContext()
    b = OperationContext()
    a.metadata["k"] = 1
    assert b.metadata == {}  # default_factory -> per-instance, not shared


def test_dynamic_attribute_assignment_still_works():
    # Third-party hooks may set arbitrary attributes.
    ctx = OperationContext()
    ctx.custom_field = "value"
    assert ctx.custom_field == "value"


def test_context_field_names_describe_scope_and_prepared_directories():
    from dataclasses import fields
    from pathlib import Path

    names = {entry.name for entry in fields(OperationContext)}
    assert {"actions", "project_containers", "is_full_project",
            "initial_runtime_state", "prepared_dirs"} <= names
    assert not {"commands", "containers", "is_full_containers",
                "runtime_state", "prepared_files"} & names
    first = OperationContext(actions=["restart", "pull"], is_full_project=False)
    second = OperationContext()
    first.prepared_dirs["app"] = Path("one")
    assert second.prepared_dirs == {}
    assert first.actions == ["restart", "pull"]
    assert first.is_full_project is False
