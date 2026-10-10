#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The extension package owns the public declarations and URL factories."""
import importlib.util

import linktools.cntr as cntr
from linktools.cntr import ext


def test_extension_exports_preserve_root_declaration_identity():
    for name in ("Integration", "Integrations", "Nginx", "Flare", "Authelia"):
        assert getattr(cntr, name) is getattr(ext, name)
    assert set(ext.__all__) == {
        "Integration", "Integrations", "Nginx", "ResolvedSite", "Flare", "Authelia",
        "load_config_url", "load_nginx_url", "load_port_url",
    }
    assert all(getattr(ext, name).__module__.startswith("linktools.cntr.ext.") for name in ext.__all__ if name != "Integrations")


def test_previous_module_path_has_no_compatibility_alias():
    assert importlib.util.find_spec("linktools.cntr.integration") is None


def test_concrete_data_types_are_not_public_exports():
    for name in ("FlareCategory", "FlareLink", "NginxSite"):
        assert not hasattr(ext, name)
        assert not hasattr(cntr, name)
