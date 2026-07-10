#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nginx-domain resolution and reverse-proxy conf writing for a container."""
from typing import TYPE_CHECKING

from linktools.core import LazyProvider
from linktools.types import MISSING

if TYPE_CHECKING:
    from typing import Any
    from linktools.types import PathType
    from ..container import BaseContainer


class NginxMixin:

    def get_nginx_domain(self: "BaseContainer", name: str = None):

        def get_domain(cfg):
            if not self.manager.containers["nginx"].enable:
                return ""
            if not cfg.get("NGINX_WILDCARD_DOMAIN", type=bool):
                return cfg.get("NGINX_ROOT_DOMAIN")
            root_domain = cfg.get("NGINX_ROOT_DOMAIN")
            if root_domain in ("_", "localhost"):
                return root_domain
            if name is None:
                return f"{self.name}.{root_domain}"
            elif name.strip() == "":
                return root_domain
            return f"{name}.{root_domain}"

        return LazyProvider(get_domain)

    def write_nginx_conf(
            self: "BaseContainer", domain: str, *,
            proxy_name: str = MISSING, proxy_domain_name: str = MISSING,
            proxy_conf: "PathType" = MISSING, proxy_url: str = MISSING,
            https_enable: bool = MISSING, waf_enable: bool = MISSING,
            auth_enable: bool = False, auth_extra: "dict[str, Any]" = MISSING,
    ):

        nginx = self.manager.containers["nginx"]
        if nginx.enable:
            nginx.write_conf(
                container=self,
                domain=domain,
                proxy_name=proxy_name,
                proxy_domain_name=proxy_domain_name,
                proxy_conf=proxy_conf,
                proxy_url=proxy_url,
                https_enable=https_enable,
                waf_enable=waf_enable,
                auth_enable=auth_enable,
                auth_extra=auth_extra,
                flush=False,
            )
