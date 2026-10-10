#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portainer container definition."""
from typing import TYPE_CHECKING

from linktools.core import ConfigField
from linktools.decorator import cached_property
from linktools.runtime import lazy_load
from linktools.cntr import BaseContainer, Flare, Nginx
from linktools.cntr.ext import Authelia, load_nginx_url, load_port_url

if TYPE_CHECKING:
    from linktools.cntr import Integrations
    from typing import Any


class Container(BaseContainer):

    @cached_property
    def configs(self) -> "dict[str, Any]":
        return dict(
            PORTAINER_TAG="alpine",
            PORTAINER_DOMAIN=Nginx.domain(self),
            PORTAINER_AUTH_ENABLE=ConfigField(cast=bool, default=True),
            PORTAINER_PORT=ConfigField(cast=int, default=9000),
        )

    @cached_property
    def integrations(self) -> "Integrations":
        return [
            Flare.container("Portainer", "docker", load_port_url(
                self, "PORTAINER_PORT",
                https=False
            )),
            Nginx.site(
                server_name=self.get_config_later("PORTAINER_DOMAIN"),
                link=Flare.public("Portainer", "docker", "Docker管理工具"),
                proxy="http://portainer:9000",
                auth=None if self.get_config("PORTAINER_AUTH_ENABLE") else False,
                auth_bypass=(r"\.(css|js)$",),
            ),
            Authelia.oidc(
                (load_nginx_url(self, "web"),),
                enabled=lazy_load(lambda: self.get_config("PORTAINER_AUTH_ENABLE")
                                  and self.get_config("NGINX_AUTH_ENABLE")),
            ),
        ]
