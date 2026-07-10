#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ContainerResolver dependency resolution (refactor spec Phase 4).

The resolver was extracted verbatim from ContainerManager; these tests lock its
pure topological-sort behavior (order, missing-dependency error, cycle error)
independently of the full manager.
"""
import pytest

from linktools.cntr.container import ContainerError
from linktools.cntr.registry.resolver import ContainerResolver


class _FakeContainer:
    def __init__(self, name, order=0, deps=()):
        self.name = name
        self.order = order
        self.dependencies = list(deps)

    def __repr__(self):
        return f"<{self.name}>"


class _FakeManager:
    def __init__(self, containers):
        self.containers = {c.name: c for c in containers}


def test_dependencies_are_ordered_before_dependents():
    a = _FakeContainer("a", order=1, deps=["b"])
    b = _FakeContainer("b", order=2)
    resolved = ContainerResolver(_FakeManager([a, b])).resolve_dependencies([a])
    assert [c.name for c in resolved] == ["b", "a"]


def test_missing_dependency_raises():
    a = _FakeContainer("a", deps=["ghost"])
    with pytest.raises(ContainerError):
        ContainerResolver(_FakeManager([a])).resolve_dependencies([a])


def test_circular_dependency_raises():
    a = _FakeContainer("a", deps=["b"])
    b = _FakeContainer("b", deps=["a"])
    with pytest.raises(ContainerError):
        ContainerResolver(_FakeManager([a, b])).resolve_dependencies([a])


def test_builtin_resolution_places_deps_first(fresh_manager):
    containers = [fresh_manager.containers[n] for n in ("authelia", "safeline", "nginx", "lldap")]
    resolved = fresh_manager.resolver.resolve_dependencies(containers)
    names = [c.name for c in resolved]
    # authelia -> [nginx, lldap]; safeline -> [nginx]
    assert names.index("nginx") < names.index("authelia")
    assert names.index("lldap") < names.index("authelia")
    assert names.index("nginx") < names.index("safeline")
