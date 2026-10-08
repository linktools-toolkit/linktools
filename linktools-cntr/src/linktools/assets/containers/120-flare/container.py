#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Flare container definition."""
from typing import TYPE_CHECKING

import yaml

from linktools import utils
from linktools.core import ConfigField, LazyProvider
from linktools.decorator import cached_property
from linktools.cntr import BaseContainer, NginxSite, ContainerError
from linktools.cntr.container import ExposeLink
from linktools.errors import ConfigNotFoundError
from linktools.rich import prompt

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Iterable
    from linktools.cntr import EventContext


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
    def integrations(self) -> "dict[str, dict[str, NginxSite]]":
        return {
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

    @cached_property
    def exposes(self) -> "Iterable[ExposeLink]":
        return [
            self.expose_container("Flare", "bookmark", "主页", self.load_port_url("FLARE_PORT", https=False)),
        ]

    def on_starting(self, context: "EventContext") -> None:

        categories = {}
        apps = []
        bookmarks = []

        for container in sorted(self.manager.installed_state.get(), key=lambda o: o.order):
            for expose in container.exposes:
                if not isinstance(expose, ExposeLink) or not expose.is_valid:
                    continue
                category = expose.category
                existing = categories.get(category.name)
                if existing is None:
                    existing = (category.desc, [])
                    categories[category.name] = existing
                elif existing[0] != category.desc:
                    raise ContainerError(
                        f"Conflicting description for category {category.name!r}")
                existing[1].append(expose)
                if category.name == "public":
                    apps.append(expose)
                bookmarks.append(expose)

        data = {"links": []}
        for app in apps:
            data["links"].append({
                "name": app.name,
                "desc": app.desc,
                "icon": app.icon,
                "link": app.url,
            })
        utils.write_file(
            self.get_app_path("app", "apps.yml", create_parent=True),
            yaml.dump(data),
        )

        data = {"categories": [], "links": []}
        for name, (description, links) in categories.items():
            if name == "public":
                continue
            if not links:
                continue
            data["categories"].append({
                "id": name,
                "title": description,
            })
            for link in links:
                data["links"].append({
                    "category": name,
                    "name": link.name,
                    "icon": link.icon,
                    "link": link.url,
                })
        utils.write_file(
            self.get_app_path("app", "bookmarks.yml", create_parent=True),
            yaml.dump(data),
        )
