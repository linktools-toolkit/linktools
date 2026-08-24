#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pytest fixtures for the cntr snapshot suite."""
import pytest

import _harness


def _close_manager_cache(manager) -> None:
    cache = vars(manager.environ).get("cache")
    if cache is not None:
        cache.close()


@pytest.fixture(scope="session")
def snapshot_manager(tmp_path_factory):
    """A fully-prepared ContainerManager over an isolated temp data dir.

    Session-scoped: building it (discovering + rendering all containers) is
    expensive, and the rendered compose is a pure function of the deterministic
    harness config, so one manager serves the whole session.
    """
    data = tmp_path_factory.mktemp("data")
    temp = tmp_path_factory.mktemp("temp")
    manager = _harness.make_manager(data, temp)
    try:
        yield manager
    finally:
        _close_manager_cache(manager)


@pytest.fixture(scope="session")
def normalize(snapshot_manager):
    """``normalize_compose`` bound to the session manager (for path scrubbing)."""
    def _normalize(data):
        return _harness.normalize_compose(data, snapshot_manager)
    return _normalize


@pytest.fixture
def fresh_manager(tmp_path):
    """A function-scoped manager that may be freely mutated.

    Distinct from ``snapshot_manager`` (session-scoped, must stay pristine for the
    snapshot regression tests). All builtins are installed and prepared.
    """
    manager = _harness.make_manager(tmp_path / "data", tmp_path / "temp")
    try:
        yield manager
    finally:
        _close_manager_cache(manager)
