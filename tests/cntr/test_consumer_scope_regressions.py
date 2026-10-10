#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Required regressions not yet closed. These tests intentionally remain gates,
not skips or expected failures, until dependency propagation is implemented.
"""
from types import SimpleNamespace

from test_lifecycle_rebuild import setup_case


def test_navigation_attached_to_nginx_site_refreshes_running_flare(tmp_path):
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:new"}}),
        ("nginx", {"nginx": {"image": "nginx:new"}}),
        ("flare", {"flare": {"image": "flare:new"}}),
    ], running=("nginx", "flare"))
    # Nginx carries an attached Flare through its public link field.
    manager.integration_snapshot["app"] = (
        SimpleNamespace(consumer="nginx", link=SimpleNamespace(consumer="flare")),
    )
    manager.compose_operations.up(["app"])
    assert ("prepare", "flare") in manager.events


def test_removing_producers_last_site_refreshes_previous_running_consumer(tmp_path):
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:new"}}),
        ("nginx", {"nginx": {"image": "nginx:new"}}),
    ], running=("nginx",))
    old = manager.containers["nginx"].get_app_path("generated/old/sites/app.conf")
    old.parent.mkdir(parents=True)
    old.write_text("# Previously published app site\nserver { server_name app.example.test; }\n")
    manager.artifact_index.record({
        str(old.relative_to(tmp_path)): {
            "kind": "generated-config", "container": "nginx",
            "sha256": "prior", "producers": ["app"],
        },
    })
    manager.integration_snapshot["app"] = ()
    manager.compose_operations.up(["app"])
    assert ("prepare", "nginx") in manager.events
