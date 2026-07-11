#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two independent ContainerManager instances (simulating two separate
`ct-cntr` processes sharing one data dir) must not lose each other's
add/remove writes to INSTALLED_CONTAINERS / INSTALLED_REPOS.

Regression: InstalledStateStore.add()/remove() and RepoStore.add()/remove()
read via self._load() (which reads whatever the process's own ConfigStore
last cached in memory) and wrote via self._dump() (ConfigStore.set(), which
reloads from disk right before flushing but then unconditionally overwrites
the key with the value computed from the earlier, possibly-stale read). The
"cntr:settings"/"cntr:repo" process lock only serialized the two writers; it
never forced either one to see the other's write before computing its own
new value, so the second writer's flush silently discarded the first
writer's change.
"""
from _harness import install_deterministic_interaction, _reset_global_config


def _two_managers(tmp_path, monkeypatch):
    install_deterministic_interaction()
    _reset_global_config()
    storage = str(tmp_path)
    monkeypatch.setenv("LINKTOOLS_PATH", storage)
    monkeypatch.setenv("LINKTOOLS_DATA_PATH", storage + "/data")
    monkeypatch.setenv("LINKTOOLS_TEMP_PATH", storage + "/temp")

    from linktools.core._environ import Environ
    from linktools.cntr.manager import ContainerManager

    # Each Environ()/ContainerManager pair gets its own ConfigStore instance
    # (its own in-memory _data cache), exactly like two separate `ct-cntr`
    # process invocations sharing the same on-disk data directory.
    manager_a = ContainerManager(Environ(), name="aio")
    manager_b = ContainerManager(Environ(), name="aio")
    return manager_a, manager_b


def test_two_processes_adding_different_containers_both_persist(tmp_path, monkeypatch):
    manager_a, manager_b = _two_managers(tmp_path, monkeypatch)
    name_a, name_b = list(manager_a.containers.keys())[:2]

    manager_a.installed_state.add(name_a)
    manager_b.installed_state.add(name_b)  # manager_b's store was cached before A's write

    # A third, freshly constructed manager reads the final on-disk state.
    from linktools.core._environ import Environ
    from linktools.cntr.manager import ContainerManager
    observer = ContainerManager(Environ(), name="aio")
    installed = set(observer.installed_state.load_names())
    assert installed == {name_a, name_b}


def test_two_processes_adding_different_repos_both_persist(tmp_path, monkeypatch, tmp_path_factory):
    manager_a, manager_b = _two_managers(tmp_path, monkeypatch)

    repo_a = tmp_path_factory.mktemp("repo_a")
    (repo_a / "container.py").write_text("# placeholder\n")
    repo_b = tmp_path_factory.mktemp("repo_b")
    (repo_b / "container.py").write_text("# placeholder\n")

    manager_a.repo_store.add(str(repo_a), force=True)
    manager_b.repo_store.add(str(repo_b), force=True)  # manager_b's store was cached before A's write

    from linktools.core._environ import Environ
    from linktools.cntr.manager import ContainerManager
    observer = ContainerManager(Environ(), name="aio")
    repos = observer.repo_store.get_all()
    assert str(repo_a) in repos
    assert str(repo_b) in repos
