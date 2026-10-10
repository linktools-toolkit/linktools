#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The extension package owns the public declarations and URL factories."""
import importlib.util

import pytest

import linktools.cntr as cntr
from linktools.cntr import ext


def test_extension_exports_preserve_generic_contract_identity() -> None:
    for name in ("Integration", "Integrations"):
        assert getattr(cntr, name) is getattr(ext, name)
    assert set(ext.__all__) == {
        "Integration", "Integrations", "Nginx", "ResolvedSite", "Flare", "Authelia",
        "load_config_url", "load_nginx_url", "load_port_url",
    }
    assert all(getattr(ext, name).__module__.startswith("linktools.cntr.ext.") for name in ext.__all__ if name != "Integrations")


@pytest.mark.parametrize("name", (
    "Nginx", "ResolvedSite", "Flare", "Authelia",
    "load_config_url", "load_nginx_url", "load_port_url",
))
def test_consumer_specific_apis_are_only_exported_from_ext(name: str) -> None:
    assert getattr(ext, name) is not None
    assert not hasattr(cntr, name)
    assert name not in cntr.__all__
    with pytest.raises(ImportError):
        exec("from linktools.cntr import " + name, {})


def test_root_wildcard_exports_only_generic_contracts() -> None:
    expected = {
        "ContainerError", "BaseContainer", "SourceContainer",
        "Integration", "Integrations", "ContainerManager", "OperationContext",
    }
    assert set(cntr.__all__) == expected
    namespace = {}
    exec("from linktools.cntr import *", namespace)
    assert set(namespace) - {"__builtins__"} == expected
    assert all(namespace[name] is getattr(cntr, name) for name in expected)


def test_previous_module_path_has_no_compatibility_alias():
    assert importlib.util.find_spec("linktools.cntr.integration") is None


def test_concrete_data_types_are_not_public_exports():
    for name in ("FlareCategory", "FlareLink", "NginxSite"):
        assert not hasattr(ext, name)
        assert not hasattr(cntr, name)
