#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Selected Authelia consumers require HTTPS; Redis-only operations do not."""

from types import SimpleNamespace

import pytest

from _harness import builtin_container_type
from linktools.cntr import ContainerError, OperationContext
from linktools.cntr._operations import ComposeOperations, ComposeSelection


@pytest.mark.parametrize("services", [
    None, (), ("authelia-redis",), ("authelia",), ("authelia-admin",),
])
def test_authelia_https_check_only_for_configuration_consumers(services):
    native = builtin_container_type("102-authelia")
    reads = []
    owner = SimpleNamespace(_config_services=native._config_services,
                            get_config=lambda key: reads.append(key) or False)
    context = OperationContext(target_services=services)
    selected = services is None or bool(set(services) & set(native._config_services))
    if selected:
        with pytest.raises(ContainerError, match="Authelia requires HTTPS"):
            native.on_check(owner, context)
        assert reads == ["NGINX_HTTPS_ENABLE"]
    else:
        native.on_check(owner, context)
        assert reads == []


def test_authelia_redis_only_does_not_prepare_configuration():
    native = builtin_container_type("102-authelia")
    def unexpected(*args, **kwargs):
        raise AssertionError("Redis must not create Authelia configuration")
    owner = SimpleNamespace(_config_services=native._config_services, get_config=unexpected)
    context = OperationContext(target_services=("authelia-redis",))
    native.on_check(owner, context)
    native.on_starting(owner, context)


@pytest.mark.parametrize("full", [False, True])
def test_context_exposes_exact_selected_services(full):
    owner = SimpleNamespace(name="app", services={"app": {}, "sidecar": {}})
    selection = ComposeSelection((owner,), (owner,), () if full else ("sidecar",), full)
    context = ComposeOperations(SimpleNamespace())._make_context("up", selection)
    assert context.target_services == (("app", "sidecar") if full else ("sidecar",))
