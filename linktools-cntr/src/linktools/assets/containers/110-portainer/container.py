#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portainer container definition."""
from typing import TYPE_CHECKING

from linktools.core import ConfigRef, ConfigLazy, ConfigField
from linktools.decorator import cached_property
from linktools.cntr import BaseContainer
from linktools.cntr.ext import endpoint, Flare, Nginx, Authelia, load_port_url

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
        return endpoint(
            self,
            domain=ConfigRef("PORTAINER_DOMAIN"),
            proxy="http://portainer:9000",
            auth=ConfigLazy(lambda config: None if config.get("PORTAINER_AUTH_ENABLE") else False),
            auth_bypass=(r"\.(css|js)$",),
            extensions=[
                Flare.links(
                    Flare.public(),
                    Flare.container(
                        url=load_port_url(self, "PORTAINER_PORT", https=False),
                    ),
                    name="Portainer",
                    icon="docker",
                    desc="Docker管理工具",
                ),
                Authelia.oidc(("{url}",)),
            ],
        )
