#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A partial start prepares needed integration providers and Compose dependencies."""


def test_partial_start_expands_required_integration_providers(fresh_manager):
    operations = fresh_manager.compose_operations
    selection = operations.select(("portainer",), metadata_only=True, for_start=True)
    assert [c.name for c in selection.target_containers] == ["portainer"]

    expanded = operations.start_selection(selection)
    names = {c.name for c in expanded.target_containers}
    assert {"nginx", "portainer", "safeline", "authelia", "lldap"} <= names
    assert not expanded.full
    assert {"nginx", "portainer", "safeline-mgt", "authelia", "lldap"} <= set(expanded.services)


def test_pending_authelia_redis_does_not_expand_owner_container_dependencies(fresh_manager):
    from types import SimpleNamespace

    operations = fresh_manager.compose_operations
    explicit = operations.select(("lldap",), metadata_only=True, for_start=True)
    context = SimpleNamespace(
        initial_running_services={"authelia-redis"},
        changed_compose_services={"authelia-redis"},
    )
    selection = operations._reconcile_selection(explicit, context)
    assert "authelia-redis" in selection.services
    assert "lldap" in selection.services
    assert "authelia" not in selection.services
    assert "nginx" not in selection.services
    assert "authelia-redis" in {
        name for container in selection.target_containers for name in container.services
    }


def test_explicit_restart_targets_remain_narrow(fresh_manager):
    operations = fresh_manager.compose_operations
    selected = operations.select(("portainer",), metadata_only=True, for_start=True)
    expanded = operations.start_selection(selected)
    assert {c.name for c in selected.target_containers} == {"portainer"}
    assert "nginx" in {c.name for c in expanded.target_containers}
    assert "nginx" not in set(selected.services)
