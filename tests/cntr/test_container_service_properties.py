#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Container direct access to shared manager services (Spec section 64/90):
each property must return the exact same instance the manager already
caches, never a second copy, and internal code must reach it directly
instead of bouncing through a removed Manager wrapper."""
from linktools.decorator import cached_property
from linktools.cntr.manager import ContainerManager
from linktools.cntr.runtime.process import RuntimeProcessFactory


def test_container_service_properties_are_manager_singletons(fresh_manager):
    container = fresh_manager.containers["nginx"]
    assert container.environ is fresh_manager.environ
    assert container.env_config is fresh_manager.env_config
    assert container.runtime is fresh_manager.runtime
    assert container.compose_runner is fresh_manager.compose_runner
    assert container.lifecycle is fresh_manager.lifecycle
    assert container.running_state is fresh_manager.running_state
    assert container.project_name == fresh_manager.project_name
    assert container.host == fresh_manager.host
    assert container.user == fresh_manager.user
    assert container.containers is fresh_manager.containers


def test_container_service_properties_never_cache_a_second_copy(fresh_manager):
    container = fresh_manager.containers["nginx"]
    assert container.runtime is container.runtime
    assert "runtime" not in container.__dict__  # plain property, not cached_property


def test_container_lacks_global_manager_management_surface(fresh_manager):
    """Section 65: global repo/state/discovery management stays manager-only;
    Container must not become a full proxy for it."""
    container = fresh_manager.containers["nginx"]
    for name in (
        "loader", "resolver", "repo_store", "installed_state",
        "_persistent_store", "_transient_ns", "_migrated",
        "add_repo", "update_repos", "remove_repo",
        "add_installed_containers", "remove_installed_containers",
    ):
        assert not hasattr(container, name), f"Container should not expose manager.{name}"


def test_custom_runtime_service_can_override_process_creation(fresh_manager):
    """A downstream ContainerManager subclass replaces manager.runtime (a
    formal service) rather than re-adding a manager.create_process wrapper
    (Spec section 68.2)."""

    class CustomRuntimeProcessFactory(RuntimeProcessFactory):
        pass

    class CustomManager(ContainerManager):
        @cached_property
        def runtime(self):
            return CustomRuntimeProcessFactory(self)

    manager = CustomManager(fresh_manager.environ, name=fresh_manager.name)
    container = manager.containers["nginx"]
    assert isinstance(container.runtime, CustomRuntimeProcessFactory)
    assert container.runtime is manager.runtime
