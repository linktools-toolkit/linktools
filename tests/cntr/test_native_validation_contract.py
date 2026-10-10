#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native validation uses the selected image and candidate mounts without exposing ports."""

from types import SimpleNamespace

from linktools.cntr.runtime.compose import ComposeRunner


def test_isolated_validation_preserves_image_env_and_mounts_without_network_identity():
    runner = ComposeRunner(SimpleNamespace(project_name="project"))
    model = {"services": {"nginx": {
        "image": "nginx:target", "ports": ["80:80"],
        "depends_on": {"app": {}},
        "networks": {"private": {"ipv4_address": "10.0.0.2"}},
        "environment": {"PASSWORD_FILE": "/generated/current/password"},
        "volumes": [{"type": "bind", "source": "/host/generated",
                     "target": "/generated", "read_only": True}],
    }}}
    args = runner.isolated_service_args(
        model, "nginx", ("nginx", "-t"),
        {"PASSWORD_FILE": "/generated/candidate/password"})
    assert args[:5] == ["run", "--rm", "--network", "none", "--mount"]
    assert "PASSWORD_FILE=/generated/candidate/password" in args
    assert "nginx:target" in args
    assert not any(value in " ".join(args)
                   for value in ("10.0.0.2", "80:80", "depends_on"))


def test_isolated_mount_override_replaces_only_certificate_volume():
    runner = ComposeRunner(SimpleNamespace(project_name="project"))
    model = {"services": {"nginx": {"image": "nginx:target", "volumes": [
        {"type": "bind", "source": "/host/certs", "target": "/etc/certs"},
        {"type": "bind", "source": "/host/generated",
         "target": "/etc/nginx/generated", "read_only": True},
    ]}}}
    args = runner.isolated_service_args(
        model, "nginx", ("nginx", "-t"),
        mount_overrides={"/etc/certs": "/host/candidate"})
    assert "type=bind,source=/host/candidate,target=/etc/certs,readonly" in args
    assert "type=bind,source=/host/generated,target=/etc/nginx/generated,readonly" in args
    assert "/host/certs" not in " ".join(args)
