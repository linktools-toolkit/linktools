# -*- coding: utf-8 -*-
"""BaseContainer.docker_compose must not hold a settings transaction open
across template rendering.

docker_compose only ever *reads* `mount_paths` (no write happens in that
property), so it needs no transaction of its own. render_template() builds a
Jinja context giving templates access to every other installed container
(`containers=self.manager.containers`); since CacheStore's transaction-nesting
guard (`_tx_owner`) is store-wide, not per-namespace, holding a transaction
open across the whole render would collide with any OTHER container's compose
template that cross-references a container property backed by its own
`with self.settings.transaction()`.

Reproduced here with a synthetic two-container repo whose compose.yml
references another container's settings-backed property mid-render.
"""
import os

import pytest

import _harness


@pytest.fixture
def cross_ref_repo(tmp_path):
    """A tiny local "repo" with one container whose compose.yml reads
    another container's settings-backed property mid-render."""
    repo = tmp_path / "repo"
    a_dir = repo / "100-a"
    b_dir = repo / "200-b"
    a_dir.mkdir(parents=True)
    b_dir.mkdir(parents=True)

    (a_dir / "container.py").write_text(
        "from linktools.cntr import BaseContainer\n"
        "class Container(BaseContainer):\n"
        "    @property\n"
        "    def settings_backed_value(self):\n"
        "        with self.settings.transaction() as settings:\n"
        "            value = settings.get('value', default=None)\n"
        "            if value is None:\n"
        "                value = 'computed'\n"
        "                settings.set('value', value)\n"
        "            return value\n"
    )
    (a_dir / "compose.yml").write_text(
        "services:\n"
        "  a:\n"
        "    image: busybox\n"
    )

    (b_dir / "container.py").write_text(
        "from linktools.cntr import BaseContainer\n"
        "class Container(BaseContainer):\n"
        "    pass\n"
    )
    (b_dir / "compose.yml").write_text(
        "services:\n"
        "  b:\n"
        "    image: busybox\n"
        "    environment:\n"
        "      - CROSS_REF={{ manager.containers['a'].settings_backed_value }}\n"
    )
    return repo


def test_cross_container_settings_access_during_docker_compose_does_not_nest(
        tmp_path, cross_ref_repo):
    # Build the manager and add the repo before `.containers` (a
    # cached_property) is ever touched, so our synthetic "a"/"b" containers
    # are actually discovered -- unlike the shared `fresh_manager` fixture,
    # which already memoizes `.containers` over just the builtins.
    _harness.install_deterministic_interaction()
    _harness._reset_global_config()
    data_path = tmp_path / "data"
    temp_path = tmp_path / "temp"
    storage = str(data_path.parent)
    os.environ["LINKTOOLS_PATH"] = storage
    os.environ["LINKTOOLS_DATA_PATH"] = str(data_path)
    os.environ["LINKTOOLS_TEMP_PATH"] = str(temp_path)

    from linktools.core._environ import Environ
    from linktools.cntr.manager import ContainerManager

    environ = Environ()
    manager = ContainerManager(environ, name="aio")
    manager.repos.add(str(cross_ref_repo))
    manager.installed_state.add("a", "b")

    # Must not raise CacheTransactionError.
    containers = manager.prepare_installed_containers()
    b = next(c for c in containers if c.name == "b")
    assert b.docker_compose["services"]["b"]["environment"] == ["CROSS_REF=computed"]
