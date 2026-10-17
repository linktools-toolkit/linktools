#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validated header mapping order does not change generated nginx identity."""
from _harness import configure_builtin_inputs, make_manager
from linktools.cntr.ext import endpoint


def test_auth_header_mapping_order_preserves_rendered_files_and_path(tmp_path):
    results = []
    for reverse in (False, True):
        root = tmp_path / str(reverse)
        manager = make_manager(root / "data", root / "temp")
        configure_builtin_inputs(manager)
        manager.config.set("DOCKER_APP_PATH", "/test/apps")
        headers = [("X-Z-Credential", "z"), ("X-A-Credential", "a")]
        producer = manager.containers["portainer"]
        producer.__dict__["integrations"] = endpoint(
            producer, domain="app.example.test", proxy="http://app:8080", https=True,
            auth=True, auth_headers=dict(reversed(headers) if reverse else headers))
        nginx = manager.containers["nginx"]
        results.append((nginx.config_path.name, nginx._rendered_site_files))
        assert list(nginx.validated_auth_headers(dict(headers))) == ["X-A-Credential", "X-Z-Credential"]
    assert results[0] == results[1]
