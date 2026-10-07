#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Declarative integrations do not depend on navigation registration."""

from linktools.cntr import NginxSite


def test_portainer_site_is_independent_of_exposes(fresh_manager):
    portainer = fresh_manager.containers["portainer"]
    baseline = len(portainer.start_hooks)

    first = portainer.integrations["nginx"]["web"]
    assert isinstance(first, NginxSite)
    assert first.proxy == "http://portainer:9000"
    assert first.auth_bypass == (r"\.(css|js)$",)
    assert first is portainer.integrations["nginx"]["web"]

    portainer.load_nginx_url("web")
    portainer.load_nginx_url("web", "settings")
    assert len(portainer.start_hooks) == baseline


def test_nginx_consumes_sites_without_exposure_side_effects(fresh_manager):
    portainer = fresh_manager.containers["portainer"]
    original_hooks = len(portainer.start_hooks)

    entries = list(fresh_manager.iter_integrations("nginx"))
    matches = [(producer, site_id, site) for producer, site_id, site in entries
               if producer is portainer and site_id == "web"]
    assert len(matches) == 1
    assert matches[0][2] is portainer.integrations["nginx"]["web"]
    assert len(portainer.start_hooks) == original_hooks
