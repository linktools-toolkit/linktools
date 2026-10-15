#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Named starts preserve explicit services; full starts include their requirements."""


def test_partial_start_does_not_expand_required_integration_providers(fresh_manager):
    operations = fresh_manager.compose_operations
    selection = operations.select(("portainer",), for_start=True)
    assert [c.name for c in selection.target_containers] == ["portainer"]

    resolved = operations.start_selection(selection)
    assert {c.name for c in resolved.target_containers} == {"portainer"}
    assert not resolved.full
    assert resolved.services == ("portainer",)


def test_full_start_includes_required_integration_providers(fresh_manager):
    operations = fresh_manager.compose_operations
    resolved = operations.start_selection(operations.select(for_start=True))
    assert resolved.full
    assert {"nginx", "portainer", "safeline", "authelia", "lldap"} <= {
        container.name for container in resolved.target_containers}
    assert {"nginx", "portainer", "safeline-mgt", "authelia", "lldap"} <= set(resolved.services)


def test_unrelated_running_sidecar_does_not_expand_selected_scope(fresh_manager):
    operations = fresh_manager.compose_operations
    explicit = operations.select(("lldap",), for_start=True)
    selection = operations.start_selection(explicit, running_services={"authelia-redis"})
    assert "lldap" in selection.services
    assert "authelia-redis" not in selection.services
    assert "authelia" not in selection.services


def test_explicit_restart_targets_remain_narrow(fresh_manager):
    operations = fresh_manager.compose_operations
    selected = operations.select(("portainer",), for_start=True)
    expanded = operations.start_selection(selected)
    assert {c.name for c in selected.target_containers} == {"portainer"}
    assert {c.name for c in expanded.target_containers} == {"portainer"}
    assert expanded.services == selected.services == ("portainer",)
    assert "nginx" not in set(selected.services)
