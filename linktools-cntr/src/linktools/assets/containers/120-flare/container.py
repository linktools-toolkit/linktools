#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Flare container definition."""
from typing import TYPE_CHECKING


from linktools.core import ConfigField, LazyProvider
from linktools.decorator import cached_property
from linktools.cntr import BaseContainer, NginxSite
from linktools.errors import ConfigNotFoundError
from linktools.rich import prompt

if TYPE_CHECKING:
    from typing import Any


class Container(BaseContainer):

    @cached_property
    def configs(self) -> "dict[str, Any]":
        return dict(
            # NGINX_WILDCARD_DOMAIN is owned by the nginx container (its own
            # `configs` declares the field's cast/provider/default); this
            # container must not redeclare it with a different default.
            FLARE_TAG="latest",
            FLARE_DOMAIN=self.get_nginx_domain(""),
            FLARE_PORT=ConfigField(cast=int, default=5000),
            FLARE_AUTH_ENABLE=ConfigField(cast=bool, default=True),
            FLARE_LOGIN_ENABLE=ConfigField(cast=bool, default=False),
            FLARE_USER=ConfigField.chain(
                LazyProvider(lambda r: self._prompt_flare_user(r), cached=True),
                default="",
            ),
            FLARE_PASSWORD=ConfigField.chain(
                LazyProvider(lambda r: self._prompt_flare_password(r), cached=True),
                default="",
            ),
        )

    def _prompt_flare_user(self, r):
        # Raise (rather than return "") when login is disabled, so the
        # enclosing ChainProvider falls through to field.default="" without
        # ever persisting it -- a plain cached=True here would otherwise
        # permanently cache "" the first time this resolves while login
        # happens to be off, and never prompt again once it's enabled.
        if not r.get("FLARE_LOGIN_ENABLE"):
            raise ConfigNotFoundError("FLARE_LOGIN_ENABLE is disabled")
        return prompt("FLARE_USER", default="admin")

    def _prompt_flare_password(self, r):
        if not r.get("FLARE_LOGIN_ENABLE"):
            raise ConfigNotFoundError("FLARE_LOGIN_ENABLE is disabled")
        return prompt("FLARE_PASSWORD")

    @cached_property
    def integrations(self) -> "dict[str, dict[str, Any]]":
        return {
            "flare": {
                "direct": self.expose_container("Flare", "bookmark", "主页", self.load_port_url("FLARE_PORT", https=False)),
            },
            "nginx": {
                "web": NginxSite(
                    server_name=self.get_config_later("FLARE_DOMAIN"),
                    proxy="http://flare:5005",
                    auth=None if self.get_config("FLARE_AUTH_ENABLE") else False,
                    auth_bypass=(r"\.(css|js)$",),
                    auth_rule={"policy": "one_factor"} if self.get_config("FLARE_AUTH_ENABLE") else None,
                ),
            },
        }
