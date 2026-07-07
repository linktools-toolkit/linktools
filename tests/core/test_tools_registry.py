# -*- coding: utf-8 -*-
"""Tests for ToolDefinition + ToolRegistry dependency validation (spec §10.5)."""
import pytest

from linktools.core._tools_registry import ToolDefinition, ToolRegistry
from linktools.errors import ToolDependencyError


def _def(name, depends_on=(), platforms=None, architectures=None):
    return ToolDefinition(name=name, depends_on=tuple(depends_on),
                          platforms=platforms, architectures=architectures)


def test_registry_validates_clean_graph():
    reg = ToolRegistry()
    reg.add(_def("java"))
    reg.add(_def("apktool", depends_on=("java",)))
    reg.add(_def("baksmali", depends_on=("java",)))
    # No exception -> valid.
    reg.validate()
    order = reg.topological_sort()
    assert order.index("java") < order.index("apktool")
    assert order.index("java") < order.index("baksmali")


def test_missing_dependency():
    reg = ToolRegistry()
    reg.add(_def("apktool", depends_on=("java",)))  # java not registered
    with pytest.raises(ToolDependencyError):
        reg.validate()


def test_self_dependency():
    reg = ToolRegistry()
    reg.add(_def("weird", depends_on=("weird",)))
    with pytest.raises(ToolDependencyError) as exc:
        reg.validate()
    assert "weird" in str(exc.value)


def test_cyclic_dependency_chain_in_error():
    reg = ToolRegistry()
    reg.add(_def("a", depends_on=("b",)))
    reg.add(_def("b", depends_on=("c",)))
    reg.add(_def("c", depends_on=("a",)))
    with pytest.raises(ToolDependencyError) as exc:
        reg.validate()
    msg = str(exc.value)
    # full chain is present
    for n in ("a", "b", "c"):
        assert n in msg


def test_duplicate_name_rejected():
    reg = ToolRegistry()
    reg.add(_def("java"))
    with pytest.raises(ToolDependencyError):
        reg.add(_def("java"))


def test_platform_availability_filter():
    reg = ToolRegistry(current_system="linux", current_arch="x86_64")
    reg.add(_def("java", platforms=("linux", "darwin")))
    reg.add(_def("macsonly", platforms=("darwin",)))
    available = reg.available()
    assert "java" in available
    assert "macsonly" not in available


def test_topological_sort_is_stable():
    # Insertion order should not change the resolved order beyond dependency
    # constraints.
    reg = ToolRegistry()
    reg.add(_def("z"))
    reg.add(_def("a"))
    reg.add(_def("m"))
    order = reg.topological_sort()
    # No dependencies -> preserved insertion order (stable).
    assert order == ["z", "a", "m"]


def test_available_excludes_unsupported_arch():
    reg = ToolRegistry(current_system="linux", current_arch="arm64")
    reg.add(_def("x86only", architectures=("x86_64",)))
    assert "x86only" not in reg.available()
