#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""remove_installed_containers correctness.

Removal must compare containers by ``container.name``, not object identity
against a set of name strings. These tests lock the intended behavior: a leaf
container
removes cleanly; a container with dependents refuses without ``--force``; ``force``
removes the container and its dependents together.
"""
import pytest

from linktools.cntr.container import ContainerError


def _installed_names(manager):
    return {c.name for c in manager.installed_state.get(resolve=False)}


def test_remove_leaf_container_succeeds(fresh_manager):
    # portainer declares no dependencies and nothing depends on it.
    removed = fresh_manager.installed_state.remove("portainer")
    assert {c.name for c in removed} == {"portainer"}
    assert "portainer" not in _installed_names(fresh_manager)


def test_remove_container_with_dependents_without_force_raises(fresh_manager):
    # authelia and safeline depend on nginx; removing nginx without force must fail.
    with pytest.raises(ContainerError):
        fresh_manager.installed_state.remove("nginx")
    # Nothing should have been removed.
    assert "nginx" in _installed_names(fresh_manager)


def test_remove_container_with_force_removes_dependents(fresh_manager):
    removed = fresh_manager.installed_state.remove("nginx", force=True)
    removed_names = {c.name for c in removed}
    # nginx plus its direct dependents (authelia, safeline) are removed together.
    assert "nginx" in removed_names
    assert "authelia" in removed_names
    assert "safeline" in removed_names
    installed = _installed_names(fresh_manager)
    assert "nginx" not in installed
    assert "authelia" not in installed
    assert "safeline" not in installed
    # Unrelated containers are untouched.
    assert "lldap" in installed
    assert "portainer" in installed
    assert "flare" in installed


def test_remove_unknown_name_is_noop(fresh_manager):
    # Names not present in discovered containers are silently skipped (legacy behavior).
    before = _installed_names(fresh_manager)
    removed = fresh_manager.installed_state.remove("does-not-exist")
    assert removed == []
    assert _installed_names(fresh_manager) == before


def test_remove_survives_a_dependency_on_a_container_that_no_longer_exists(fresh_manager, monkeypatch):
    """Regression: is_depend_on() indexed manager.containers[next_name] directly.
    If an installed container's dependency chain names a container whose
    defining repo has since been removed (so it's no longer in
    manager.containers at all), that raised a bare KeyError -- crashing
    `remove` for an entirely unrelated container, blocking the very command a
    user would reach for to clean up that broken state.
    """
    # flare declares no real dependencies; portainer is unrelated to it too --
    # picking two containers with no real dependency edge isolates the
    # assertion to the dangling-dependency handling itself.
    flare = fresh_manager.containers["flare"]
    monkeypatch.setattr(type(flare), "dependencies", property(lambda self: ["ghost-container"]))

    removed = fresh_manager.installed_state.remove("portainer")
    assert {c.name for c in removed} == {"portainer"}
