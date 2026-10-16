#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Partial application follows current declarations and retained provenance."""
from linktools.core import ConfigLazy
from linktools.cntr.ext import ResolvedSite
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from linktools.cntr import ContainerManager
from linktools.cntr.artifacts import stage_files
from linktools.cntr.ext import Flare, Nginx, endpoint, load_nginx_url
from linktools.runtime import lazy_load
from test_lifecycle_rebuild import setup_case


def fail() -> None:
    raise AssertionError("consumer scope must not resolve lazy declaration values")


def set_declarations(manager: SimpleNamespace, producer: str, declarations: object) -> None:
    manager.containers[producer].integrations = declarations
    normalized = object.__new__(ContainerManager)
    normalized.__dict__["containers"] = manager.containers
    for owner in manager.containers.values():
        if not hasattr(owner, "integrations"):
            owner.integrations = ()
    normalized.__dict__["installed_state"] = SimpleNamespace(
        get=lambda resolve: tuple(manager.containers.values()))
    manager.integration_snapshot = normalized.integration_snapshot


@pytest.mark.parametrize("nested_navigation", [False, True])
@pytest.mark.parametrize("nested_endpoint", [False, True])
def test_independent_navigation_and_named_site_preserve_running_consumers(
        tmp_path: Path, nested_navigation: bool, nested_endpoint: bool) -> None:
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:new"}}),
        ("nginx", {"nginx": {"image": "nginx:new"}}),
        ("flare", {"flare": {"image": "flare:new"}}),
        ("other", {"other": {"image": "other:new"}}),
    ], running=("nginx", "flare", "other"))
    app = manager.containers["app"]
    navigation = Flare.public("App", "web", "Application", load_nginx_url(ConfigLazy(lambda config: ResolvedSite.find(app, None))))
    navigation = (navigation,) if nested_navigation else navigation
    declarations = ([endpoint(app, domain=lazy_load(fail), extensions=[(navigation,)])]
                    if nested_endpoint else [Nginx.site(lazy_load(fail)), navigation])
    set_declarations(manager, "app", declarations)
    assert isinstance(manager.integration_snapshot["app"][0], Nginx)
    manager.compose_operations.up(["app"])
    assert ("prepare", "nginx") not in manager.events
    assert ("prepare", "flare") not in manager.events
    assert ("prepare", "other") not in manager.events
    assert {event[1] for event in manager.events if event[0] == "apply"} == {"app"}


def test_removing_producers_last_site_preserves_unselected_running_consumer(tmp_path: Path) -> None:
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:new"}}),
        ("nginx", {"nginx": {"image": "nginx:new"}}),
    ], running=("nginx",))
    set_declarations(manager, "app", [Nginx.site(lazy_load(fail))])
    nginx = manager.containers["nginx"]
    nginx.runtime = SimpleNamespace(chmod=lambda path, mode: os.chmod(path, mode))
    old = stage_files(nginx, {"sites/app.conf": "server { server_name app.example.test; }\n"})
    assert (old / "sites/app.conf").is_file()
    entry = manager.artifact_index.load()[str((old / "sites/app.conf").relative_to(tmp_path))]
    assert entry["producers"] == ["app"]
    set_declarations(manager, "app", [])
    assert manager.integration_snapshot["app"] == ()
    manager.compose_operations.up(["app"])
    assert ("prepare", "nginx") not in manager.events
