#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integration declarations, URL references and the container consumer protocol."""
from ._authelia import Authelia
from ._base import Integration, Integrations
from ._flare import Flare
from ._nginx import Nginx
from ._nginx_site import ResolvedSite
from ._urls import load_config_url, load_nginx_url, load_port_url

__all__ = (
    "Integration", "Integrations", "Authelia",
    "Nginx", "ResolvedSite",
    "Flare",
    "load_config_url", "load_nginx_url", "load_port_url",
)
