#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A partial start prepares needed integration providers and Compose dependencies."""


def test_partial_start_expands_required_integration_providers(fresh_manager):
    operations = fresh_manager.compose_operations
    selection = operations.select(("portainer",), metadata_only=True, for_start=True)
    assert [c.name for c in selection.target_containers] == ["portainer"]

    expanded = operations._start_selection(selection)
    names = {c.name for c in expanded.target_containers}
    assert {"nginx", "portainer", "safeline", "authelia", "lldap"} <= names
    assert not expanded.full
    assert {"nginx", "portainer", "safeline-mgt", "authelia", "lldap"} <= set(expanded.services)


def test_explicit_restart_targets_remain_narrow(fresh_manager):
    operations = fresh_manager.compose_operations
    selected = operations.select(("portainer",), metadata_only=True, for_start=True)
    expanded = operations._start_selection(selected)
    assert {c.name for c in selected.target_containers} == {"portainer"}
    assert "nginx" in {c.name for c in expanded.target_containers}
    assert "nginx" not in set(selected.services)
