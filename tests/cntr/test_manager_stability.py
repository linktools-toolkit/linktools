#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verifies the stable public contract of ContainerManager: its public
surface, configs override dispatch, migration retry semantics, and
persistent vs transient store routing, so internal reorganization cannot
silently change any of them.
"""
import inspect
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from linktools.cntr.manager import ContainerManager
from linktools.decorator import _CachedProperty


class _FakeLogger:
    def __getattr__(self, _name):
        return lambda *a, **kw: None


class _FakeEnvConfig:
    def update_defaults(self, **kwargs):
        pass

    def cast(self, value, type=None):
        if type == "path":
            return os.path.abspath(os.path.expanduser(str(value)))
        return value


class _FakeEnviron:
    name = "test-env"

    def get_logger(self, _name):
        return _FakeLogger()

    def wrap_config(self, namespace=None, env_prefix=None, local_root=None):
        return _FakeEnvConfig()

    def shared_config_sources(self, namespace, env_prefix=""):
        return (None, None, None)

    def build_config(self, schema, shared_sources, local_root=None):
        return _FakeEnvConfig()

    def get_data_path(self, *parts):
        # configs property builds str(self.data_path.joinpath(...)) defaults.
        return Path(tempfile.gettempdir())


def test_default_name_uses_environ_name_when_none():
    mgr = ContainerManager(environ=_FakeEnviron(), name=None)
    assert mgr.name == "test-env"


def test_default_name_uses_environ_name_when_empty():
    mgr = ContainerManager(environ=_FakeEnviron(), name="")
    assert mgr.name == "test-env"


def test_explicit_name_wins():
    mgr = ContainerManager(environ=_FakeEnviron(), name="custom")
    assert mgr.name == "custom"


def test_change_file_mode_probes_chmod_not_chown(monkeypatch):
    calls = []

    def fake_which(cmd):
        calls.append(cmd)
        return None  # force early return so only the probe is observed

    monkeypatch.setattr(shutil, "which", fake_which)

    mgr = ContainerManager(environ=_FakeEnviron(), name="x")
    mgr.system = "linux"  # _is_chown_supported is linux-only

    with tempfile.NamedTemporaryFile() as fp:
        mgr.runtime.chmod(fp.name, mode=0o755)

    assert "chmod" in calls
    assert "chown" not in calls


# -- ContainerManager public surface -----------------------------------------

MANAGER_API = (
    "containers",
    "debug", "prepare_installed_containers",
    "project_name", "hooks", "start_hooks", "stop_hooks",
    "user", "uid", "gid", "system", "machine", "host",
    "container_type", "container_host",
    "docker_container_name", "docker_compose_names",
    "root_path", "app_path", "app_data_path",
    "data_path", "temp_path", "setting_path",
    "env_config", "compose_runner", "compose_operations",
    "resolver", "loader", "runtime",
    "lifecycle", "running_state", "installed_state", "repo_store",
)

# Manager forwarding wrappers deliberately removed: a breaking change with
# no compatibility alias. Each has a formal service entry point instead --
# see test_manager_wrapper_forwarding_methods_removed.
_REMOVED_MANAGER_WRAPPERS = (
    "create_process", "create_docker_process", "create_docker_compose_process",
    "change_file_owner", "change_file_mode",
    "notify_start", "notify_stop", "notify_remove",
    "get_installed_containers", "resolve_depend_containers",
    "add_installed_containers", "remove_installed_containers",
    "get_all_repos", "add_repo", "update_repos", "remove_repo",
    "get_running_containers", "_load_running_containers", "_dump_running_containers",
    "_callback",
    # Existed only to back Command._make_context's compatibility wrapper
    # (also removed); no formal caller ever needed it directly.
    "create_event_context",
    "lock_store",
)

# name -> (descriptor type, [(param name, kind, default)])
_METHOD_SIGNATURES = {}

# name -> expected class-level descriptor type
_DESCRIPTOR_TYPES = {
    "debug": property,
    "configs": property,
    "container_type": _CachedProperty,
    "container_host": _CachedProperty,
    "host": _CachedProperty,
    "project_name": _CachedProperty,
    "root_path": _CachedProperty,
    "app_path": _CachedProperty,
    "app_data_path": _CachedProperty,
    "data_path": _CachedProperty,
    "temp_path": _CachedProperty,
    "setting_path": _CachedProperty,
    "containers": _CachedProperty,
    "hooks": _CachedProperty,
    "start_hooks": _CachedProperty,
    "stop_hooks": _CachedProperty,
    "compose_runner": _CachedProperty,
    "compose_operations": _CachedProperty,
    "resolver": _CachedProperty,
    "loader": _CachedProperty,
    "runtime": _CachedProperty,
    "lifecycle": _CachedProperty,
    "running_state": _CachedProperty,
    "installed_state": _CachedProperty,
    "repo_store": _CachedProperty,
}


def test_manager_api_surface_present(fresh_manager):
    for name in MANAGER_API:
        assert hasattr(fresh_manager, name), f"ContainerManager is missing {name}"


def test_manager_wrapper_forwarding_methods_removed(fresh_manager):
    """These one-line delegating wrappers are deliberately
    removed with no compatibility alias; downstream must call the formal
    service instead (manager.runtime, manager.lifecycle, manager.resolver,
    manager.installed_state, manager.repo_store)."""
    for name in _REMOVED_MANAGER_WRAPPERS:
        assert not hasattr(fresh_manager, name), f"{name} should have been removed from ContainerManager"


def test_manager_method_signatures_unchanged():
    for name, expected_params in _METHOD_SIGNATURES.items():
        method = getattr(ContainerManager, name)
        params = [p for p in inspect.signature(method).parameters.values() if p.name != "self"]
        # None of these methods have a **kwargs/*args tail beyond what's
        # listed, so the param count itself must match exactly -- zip() would
        # silently ignore a dropped trailing parameter otherwise.
        assert len(params) == len(expected_params), (
            f"{name}: expected {len(expected_params)} params, got {len(params)}"
        )
        for (exp_name, exp_kind, exp_default), param in zip(expected_params, params):
            assert param.name == exp_name, f"{name}: param order changed at {param.name!r}"
            assert param.kind == exp_kind, f"{name}.{param.name}: kind changed"
            assert param.default == exp_default, f"{name}.{param.name}: default changed"


def test_manager_descriptor_types_unchanged():
    for name, expected_type in _DESCRIPTOR_TYPES.items():
        descriptor = ContainerManager.__dict__.get(name)
        assert isinstance(descriptor, expected_type), f"{name} is no longer a {expected_type.__name__}"


def test_manager_configs_override_is_dispatched_through_property(tmp_path, monkeypatch):
    """__init__ must call self.configs (not a module-level builder directly),
    so a subclass overriding ``configs`` still has its extra fields registered.
    """
    import _harness

    _harness.install_deterministic_interaction()
    _harness._reset_global_config()
    monkeypatch.setenv("LINKTOOLS_PATH", str(tmp_path))
    monkeypatch.setenv("LINKTOOLS_DATA_PATH", str(tmp_path / "data"))
    monkeypatch.setenv("LINKTOOLS_TEMP_PATH", str(tmp_path / "temp"))

    from linktools.core._environ import Environ

    class CustomManager(ContainerManager):
        @property
        def configs(self):
            configs = super().configs
            configs["CUSTOM_KEY"] = "custom-value"
            return configs

    mgr = CustomManager(Environ(), name="contract-test")
    assert mgr.env_config.get("CUSTOM_KEY") == "custom-value"


# -- Migration retry ----------------------------------------------------------

def test_migrated_failure_is_not_cached_and_retries(fresh_manager, monkeypatch):
    """A cached_property does not cache a raised exception (linktools.decorator
    ._CachedProperty only stores the result on success), so a transient failure
    in the one-time legacy migration must be retried on the next access rather
    than being permanently "stuck" failed.
    """
    calls = []
    descriptor = ContainerManager.__dict__["_migrated"]
    real_migrate = descriptor.func

    def flaky(self):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("transient failure")
        return real_migrate(self)

    # Clear whatever _migrated already cached during fixture setup, then patch
    # the underlying function so the next access fails once, then succeeds.
    fresh_manager.__dict__.pop("_migrated", None)
    monkeypatch.setattr(descriptor, "func", flaky)

    with pytest.raises(RuntimeError):
        fresh_manager._migrated

    # Retried on next access: succeeds and is now cached.
    assert fresh_manager._migrated is True
    assert fresh_manager.__dict__["_migrated"] is True
    assert len(calls) == 2


# -- Persistent vs transient store routing ------------------------------------

def test_installed_containers_route_through_persistent_store(fresh_manager, monkeypatch):
    calls = []
    monkeypatch.setattr(
        fresh_manager._persistent_store, "set",
        lambda key, value: (calls.append(key), None)[1],
    )
    fresh_manager.installed_state.add(*list(fresh_manager.containers.keys())[:1])
    assert "INSTALLED_CONTAINERS" in calls


def test_installed_repos_route_through_persistent_store(fresh_manager, monkeypatch):
    calls = []
    monkeypatch.setattr(
        fresh_manager._persistent_store, "get",
        lambda key, default=None: (calls.append(key), default)[1],
    )
    fresh_manager.repo_store.get_all()
    assert "INSTALLED_REPOS" in calls


def test_running_containers_route_through_transient_namespace(fresh_manager, monkeypatch):
    calls = []
    monkeypatch.setattr(
        fresh_manager._transient_ns, "get",
        lambda key, default=None: (calls.append(key), default)[1],
    )
    fresh_manager.running_state.get_persisted()
    assert "RUNNING_CONTAINERS" in calls


def test_generic_setting_cache_and_routing_are_gone(fresh_manager):
    for name in ("_setting_cache", "_load_setting", "_dump_setting", "_repo_path"):
        assert not hasattr(fresh_manager, name), f"{name} should have been removed from ContainerManager"


# -- manager.containers must not recurse through installed-state -------------

def test_containers_property_does_not_recurse(fresh_manager, monkeypatch):
    """manager.containers must resolve installed *names* without reading
    manager.containers again (that would be infinite recursion through
    InstalledStateStore, whose _load() maps names back to manager.containers).
    """
    calls = []
    real_load_names = fresh_manager.installed_state.load_names

    def counting_load_names():
        calls.append(1)
        assert "containers" not in fresh_manager.__dict__, (
            "manager.containers must not be re-entered while it is still being computed"
        )
        return real_load_names()

    monkeypatch.setattr(fresh_manager.installed_state, "load_names", counting_load_names)
    fresh_manager.__dict__.pop("containers", None)
    containers = fresh_manager.containers
    assert isinstance(containers, dict) and containers
    assert calls == [1]
