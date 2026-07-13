#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Loader load failures surface as structured ContainerLoadError entries
(review P2-08), not just a log-only warning that leaves callers unable to
tell "not installed" apart from "failed to load".
"""
import os

import _harness
from linktools.cntr.registry.loader import ContainerLoadError, ContainerLoadResult


def _fresh_standalone_manager(tmp_path):
    _harness.install_deterministic_interaction()
    _harness._reset_global_config()
    data_path = tmp_path / "data"
    temp_path = tmp_path / "temp"
    os.environ["LINKTOOLS_PATH"] = str(tmp_path)
    os.environ["LINKTOOLS_DATA_PATH"] = str(data_path)
    os.environ["LINKTOOLS_TEMP_PATH"] = str(temp_path)

    from linktools.core._environ import Environ
    from linktools.cntr.manager import ContainerManager

    return ContainerManager(Environ(), name="aio")


def _repo_with_broken_container(tmp_path):
    repo = tmp_path / "repo"
    c_dir = repo / "100-broken"
    c_dir.mkdir(parents=True)
    (c_dir / "container.py").write_text("raise ImportError('boom')\n", encoding="utf-8")
    return repo


def test_load_all_returns_structured_result(fresh_manager):
    result = fresh_manager.loader.load_all()
    assert isinstance(result, ContainerLoadResult)
    assert result.errors == []
    assert len(result.containers) > 0


def test_broken_container_surfaces_as_structured_load_error(tmp_path):
    repo = _repo_with_broken_container(tmp_path)
    manager = _fresh_standalone_manager(tmp_path)
    manager.repos.add(str(repo))

    result = manager.loader.load_all()

    assert len(result.errors) == 1
    error = result.errors[0]
    assert isinstance(error, ContainerLoadError)
    assert "boom" in error.message
    assert error.expected_name == "broken"


def test_installed_container_that_fails_to_load_is_reported_not_silently_skipped(tmp_path, caplog):
    repo = _repo_with_broken_container(tmp_path)
    manager = _fresh_standalone_manager(tmp_path)
    manager.repos.add(str(repo))
    # Persist the name directly -- it can never be discovered via the normal
    # `installed_state.add("broken")` path since the container never
    # successfully loads in the first place (this simulates a container
    # that loaded fine once, was installed, and later broke).
    manager.installed_state._dump_names(["broken"])

    import logging
    with caplog.at_level(logging.WARNING, logger=manager.logger.name):
        containers = manager.containers

    assert "broken" not in containers
    assert any("failed to load" in record.message for record in caplog.records)
