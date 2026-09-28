# -*- coding: utf-8 -*-
"""BaseContainer.settings must live in a persistent, non-swept store.

Regression: settings used to be a cache.namespace(...) -- convenient, but
cache/ is exactly what clean_temp_files age-sweeps as regenerable, and real
user configuration lives here (e.g. ct-cntr mount's persisted mount_paths).
An unused-for-N-days sweep must never silently drop that. settings now
lives in manager.settings (cntr.json, a dedicated ConfigStore separate
from both cache and environ's own settings.json used for ordinary resolved
config values), with a one-time migration for anything a pre-existing
installation already persisted under the old cache namespace.
"""


def test_settings_is_not_backed_by_cache(fresh_manager):
    nginx = fresh_manager.containers["nginx"]
    nginx.settings.set("k", "v")
    # Never touches the cache store at all.
    assert nginx.environ.cache.namespace("cntr:app:nginx").get("k") is None


def test_settings_survives_a_cache_sweep(fresh_manager):
    nginx = fresh_manager.containers["nginx"]
    nginx.settings.set("mount_paths", {"web": {"/data": "/host/data:/data:rw"}})

    fresh_manager.environ.clean_temp_files(expire_days=0)  # sweep everything cache-side

    assert nginx.settings.get("mount_paths") == {"web": {"/data": "/host/data:/data:rw"}}


def _reload_container(manager, container):
    """A fresh instance of ``container`` (simulating the next CLI invocation),
    whose .settings has never been accessed yet -- same pattern as
    test_authelia_oidc_clients.py::test_oidc_clients_reloaded_from_store_is_a_set_again."""
    from linktools.cntr.registry.loader import ContainerLoader
    from linktools.cntr.repo.context import RepositoryConfigContext

    builtin_context = RepositoryConfigContext(root_path=None, file_config=None, url=None, builtin=True)
    fresh_containers = list(ContainerLoader(manager)._load_one(container.root_path, builtin_context))
    return next(c for c in fresh_containers if c.name == container.name)


def test_pre_existing_cache_data_is_migrated_on_first_access(fresh_manager):
    nginx = fresh_manager.containers["nginx"]
    old_ns = nginx.environ.cache.namespace("cntr:app:nginx")
    old_ns.set("mount_paths", {"web": {"/data": "/host/data:/data:rw"}})

    fresh = _reload_container(fresh_manager, nginx)

    assert fresh.settings.get("mount_paths") == {"web": {"/data": "/host/data:/data:rw"}}
    # Migrated out of cache -- not left duplicated in both places.
    assert old_ns.get("mount_paths") is None


def test_migration_never_overwrites_existing_persistent_data(fresh_manager):
    nginx = fresh_manager.containers["nginx"]
    nginx.settings.set("mount_paths", {"web": {"/new": "/host/new:/new:rw"}})

    old_ns = nginx.environ.cache.namespace("cntr:app:nginx")
    old_ns.set("mount_paths", {"web": {"/stale": "/host/stale:/stale:rw"}})

    fresh = _reload_container(fresh_manager, nginx)
    assert fresh.settings.get("mount_paths") == {"web": {"/new": "/host/new:/new:rw"}}
