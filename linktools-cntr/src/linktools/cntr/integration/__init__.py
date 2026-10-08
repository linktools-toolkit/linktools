#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integration declarations and the container consumer protocol."""
from ._base import Integration, Integrations, IntegrationConsumer
from ._flare import Flare, FlareCategory, FlareLink
from ._nginx import Nginx, NginxSite
from ._nginx_site import ResolvedSite

__all__ = (
    "Integration", "Integrations", "IntegrationConsumer",
    "Nginx", "NginxSite", "ResolvedSite",
    "Flare", "FlareCategory", "FlareLink",
)
