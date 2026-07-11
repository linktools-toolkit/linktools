#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : deploy.py 
@time    : 2023/05/21
@site    :  
@software: PyCharm 

              ,----------------,              ,---------,
         ,-----------------------,          ,"        ,"|
       ,"                      ,"|        ,"        ,"  |
      +-----------------------+  |      ,"        ,"    |
      |  .-----------------.  |  |     +---------+      |
      |  |                 |  |  |     | -==----'|      |
      |  | $ sudo rm -rf / |  |  |     |         |      |
      |  |                 |  |  |/----|`---=    |      |
      |  |                 |  |  |   ,/|==== ooo |      ;
      |  |                 |  |  |  // |(((( [33]|    ,"
      |  `-----------------'  |," .;'| |((((     |  ,"
      +-----------------------+  ;;  | |         |,"
         /_)______________(_/  //'   | +---------+
    ___________________________/___  `,
   /  oooooooooooooooo  .o.  oooo /,   `,"-----------
  / ==ooooooooooooooo==.o.  ooo= //   ,``--{)B     ,"
 /_==__==========__==_ooo__ooo=_/'   /___________,"
"""
from typing import TYPE_CHECKING

import yaml

from linktools import utils
from linktools.core import ConfigField, LazyProvider
from linktools.decorator import cached_property
from linktools.cntr import BaseContainer
from linktools.cntr.container import ExposeMixin, ExposeLink, ExposeCategory
from linktools.rich import prompt

if TYPE_CHECKING:
    from collections.abc import Iterable
    from linktools.cntr import EventContext


class Container(BaseContainer):

    @cached_property
    def configs(self):
        return dict(
            NGINX_WILDCARD_DOMAIN=True,
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
            raise LookupError("FLARE_LOGIN_ENABLE is disabled")
        return prompt("FLARE_USER", default="admin")

    def _prompt_flare_password(self, r):
        if not r.get("FLARE_LOGIN_ENABLE"):
            raise LookupError("FLARE_LOGIN_ENABLE is disabled")
        return prompt("FLARE_PASSWORD")

    @cached_property
    def exposes(self) -> "Iterable[ExposeLink]":
        return [
            self.expose_container("Flare", "bookmark", "主页", self.load_port_url("FLARE_PORT", https=False)),
        ]

    def on_starting(self, context: "EventContext"):

        categories = {}
        apps = []
        bookmarks = []

        for key, value in vars(ExposeMixin).items():
            if isinstance(value, ExposeCategory):
                categories.setdefault(value, list())

        for container in sorted(self.manager.installed_state.get(), key=lambda o: o.order):
            for expose in container.exposes:
                if isinstance(expose, ExposeLink) and expose.is_valid:
                    categories[expose.category].append(expose)
                    if expose.category is self.expose_public:
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
        for category, links in categories.items():
            if category.name == "public":
                continue
            if not links:
                continue
            data["categories"].append({
                "id": category.name,
                "title": category.desc,
            })
            for link in links:
                data["links"].append({
                    "category": category.name,
                    "name": link.name,
                    "icon": link.icon,
                    "link": link.url,
                })
        utils.write_file(
            self.get_app_path("app", "bookmarks.yml", create_parent=True),
            yaml.dump(data),
        )

        self.write_nginx_conf(
            domain=self.get_config("FLARE_DOMAIN"),
            proxy_url="http://flare:5005",
            auth_enable=self.get_config("FLARE_AUTH_ENABLE"),
            auth_extra={
                "acl_bypass": ["\\.(css|js)$"],
                "acl_rule": {
                    "policy": "one_factor",
                }
            }
        )
