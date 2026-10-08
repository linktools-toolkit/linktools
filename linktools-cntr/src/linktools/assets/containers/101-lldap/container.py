#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLDAP container definition."""
from typing import TYPE_CHECKING

from linktools import utils
from linktools.cntr import BaseContainer, ContainerError
from linktools.core import ConfigField, PromptProvider, LazyProvider
from linktools.decorator import cached_property

if TYPE_CHECKING:
    from typing import Any
    from linktools.cntr import EventContext


class Container(BaseContainer):

    @cached_property
    def configs(self) -> "dict[str, Any]":
        def get_base_dn(cfg: "dict[str, Any]") -> str:
            domain = cfg.get("NGINX_ROOT_DOMAIN")
            parts = domain.split(".")
            return ",".join([f"dc={part}" for part in parts])

        return dict(
            LLDAP_TAG="stable",
            LLDAP_DOMAIN=self.get_nginx_domain("ldap"),
            LLDAP_PORT=ConfigField(cast=int, default=0),
            LLDAP_WEB_PORT=ConfigField(cast=int, default=0),
            LLDAP_BASE_DN=ConfigField(provider=LazyProvider(lambda r: get_base_dn(r))),
            LLDAP_ADMIN_PASSWORD=ConfigField(provider=PromptProvider(
                default=utils.random_string(20), cached=True,
            )),
        )

    @cached_property
    def integrations(self) -> "dict[str, dict[str, Any]]":
        return {
            "flare": {
                "web": self.expose_container("LDAP", "account", "账号管理", self.load_port_url(
                    "LLDAP_WEB_PORT",
                    https=False,
                )),
            },
        }

    def on_check(self, context: "EventContext") -> None:
        domain = self.get_config("NGINX_ROOT_DOMAIN")
        if not domain or "." not in domain:
            raise ContainerError(f"Invalid domain `{domain}` for LDAP, "
                                 f"Please set NGINX_ROOT_DOMAIN to a valid domain (e.g., example.com).")
