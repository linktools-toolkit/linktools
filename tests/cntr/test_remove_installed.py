#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""remove_installed_containers correctness (refactor spec §5.2).

The old code compared a container object against a set of name strings
(``container not in remove_names``), which is always true. Removal must compare
by ``container.name``. These tests lock the intended behavior: a leaf container
removes cleanly; a container with dependents refuses without ``--force``; ``force``
removes the container and its dependents together.
"""
import pytest

from linktools.cntr.container import ContainerError


def _installed_names(manager):
    return {c.name for c in manager.get_installed_containers(resolve=False)}


def test_remove_leaf_container_succeeds(fresh_manager):
    # portainer declares no dependencies and nothing depends on it.
    removed = fresh_manager.remove_installed_containers("portainer")
    assert {c.name for c in removed} == {"portainer"}
    assert "portainer" not in _installed_names(fresh_manager)


def test_remove_container_with_dependents_without_force_raises(fresh_manager):
    # authelia and safeline depend on nginx; removing nginx without force must fail.
    with pytest.raises(ContainerError):
        fresh_manager.remove_installed_containers("nginx")
    # Nothing should have been removed.
    assert "nginx" in _installed_names(fresh_manager)


def test_remove_container_with_force_removes_dependents(fresh_manager):
    removed = fresh_manager.remove_installed_containers("nginx", force=True)
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
    removed = fresh_manager.remove_installed_containers("does-not-exist")
    assert removed == []
    assert _installed_names(fresh_manager) == before
