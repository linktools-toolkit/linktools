#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Builtin compose snapshot regression tests (refactor spec Phase 0).

Each builtin container's normalized ``docker_compose`` must match its committed
snapshot. Any drift means the refactor changed generated compose output and must
be explained (refactor spec §4.5 / §17.1).
"""
import _harness

BUILTIN_CONTAINERS = ["nginx", "lldap", "authelia", "safeline", "portainer", "flare"]


def _snapshot_path(name):
    from pathlib import Path
    return Path(__file__).parent / "snapshots" / "builtin" / f"{name}.compose.json"


def test_all_builtins_discovered(snapshot_manager):
    discovered = set(snapshot_manager.containers.keys())
    missing = [n for n in BUILTIN_CONTAINERS if n not in discovered]
    assert not missing, f"expected builtin containers missing: {missing}"


def _check(snapshot_manager, name):
    container = snapshot_manager.containers[name]
    actual = _harness.normalize_compose(container.docker_compose, snapshot_manager)
    expected = _snapshot_path(name).read_text(encoding="utf-8")
    assert actual == expected, (
        f"compose snapshot drift for builtin container `{name}`.\n"
        f"Regenerate with: python scripts/cntr_generate_snapshots.py\n"
        f"Only commit the change if the drift is intentional and explained."
    )


def test_nginx_snapshot(snapshot_manager):
    _check(snapshot_manager, "nginx")


def test_lldap_snapshot(snapshot_manager):
    _check(snapshot_manager, "lldap")


def test_authelia_snapshot(snapshot_manager):
    _check(snapshot_manager, "authelia")


def test_safeline_snapshot(snapshot_manager):
    _check(snapshot_manager, "safeline")


def test_portainer_snapshot(snapshot_manager):
    _check(snapshot_manager, "portainer")


def test_flare_snapshot(snapshot_manager):
    _check(snapshot_manager, "flare")
