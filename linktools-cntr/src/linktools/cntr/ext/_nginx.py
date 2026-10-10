#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nginx declarations, lazy domain configuration, and site resolution."""
from typing import TYPE_CHECKING

from ._base import Integration

if TYPE_CHECKING:
    from typing import Any, Mapping, Optional, Sequence
    from linktools.core import ConfigResolver, LazyProvider
    from linktools.types import PathType
    from ..container import BaseContainer
    from ._flare import Flare


class Nginx(Integration):
    """One nginx site owned by a container and identified by its local ID."""

    consumer = "nginx"
    requires_local_id = True

    @classmethod
    def site(
            cls,
            server_name: str,
            proxy: "Optional[str]" = None,
            template: "Optional[PathType]" = None,
            https: "Optional[bool]" = None,
            waf: "Optional[bool]" = None, waf_bypass: "Sequence[str]" = (),
            auth: "Optional[bool]" = None, auth_bypass: "Sequence[str]" = (),
            auth_headers: "Optional[Mapping[str, str]]" = None,
            auth_rule: "Optional[Mapping[str, Any]]" = None,
            url: "Optional[str]" = None,
            cert_domains: "Sequence[str]" = (),
            vars: "Optional[Mapping[str, Any]]" = None,
            expose: "Optional[Flare]" = None,
            *,
            local_id: str = "web",
            default: bool = False,
    ) -> "Nginx":
        self = cls()
        self.local_id = local_id
        self.expose = expose
        self.server_name = server_name
        self.default = default
        self.proxy = proxy
        self.template = template
        self.https = https
        self.waf = waf
        self.auth = auth
        self.waf_bypass = tuple(waf_bypass) if type(waf_bypass) in (tuple, list) else waf_bypass
        self.auth_bypass = tuple(auth_bypass) if type(auth_bypass) in (tuple, list) else auth_bypass
        self.auth_headers = {} if auth_headers is None else auth_headers
        self.auth_rule = auth_rule
        self.url = url
        self.cert_domains = tuple(cert_domains) if type(cert_domains) in (tuple, list) else cert_domains
        self.vars = {} if vars is None else vars
        return self

    @classmethod
    def domain(cls, container: "BaseContainer", name: "Optional[str]" = None) -> "LazyProvider":
        """Resolve a container's domain from the installed nginx configuration."""
        from linktools.core import LazyProvider

        def get_domain(config: "ConfigResolver") -> str:
            if not container.containers["nginx"].enable:
                return ""
            if not config.get("NGINX_WILDCARD_DOMAIN", type=bool):
                return config.get("NGINX_ROOT_DOMAIN")
            root = config.get("NGINX_ROOT_DOMAIN")
            if root in ("_", "localhost"):
                return root
            if name is None:
                return f"{container.name}.{root}"
            if name.strip() == "":
                return root
            return f"{name}.{root}"

        return LazyProvider(get_domain)

