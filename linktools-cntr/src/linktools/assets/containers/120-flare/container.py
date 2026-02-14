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
from typing import Iterable

import yaml

from linktools import utils
from linktools.core import Config
from linktools.decorator import cached_property
from linktools.cntr import BaseContainer
from linktools.cntr.container import ExposeMixin, ExposeLink, ExposeCategory


class Container(BaseContainer):

    @cached_property
    def configs(self):
        return dict(
            NGINX_WILDCARD_DOMAIN=True,
            FLARE_TAG="latest",
            FLARE_DOAMIN=self.get_nginx_domain(""),
            FLARE_EXPOSE_PORT=Config.Property(type=int) | 0,
            FLARE_ENABLE_LOGIN=Config.Confirm(cached=True) | False,
            FLARE_USER=Config.Lazy(
                lambda cfg:
                Config.Prompt(cached=True) | "admin"
                if cfg.get("FLARE_ENABLE_LOGIN")
                else ""
            ),
            FLARE_PASSWORD=Config.Lazy(
                lambda cfg:
                Config.Prompt(cached=True)
                if cfg.get("FLARE_ENABLE_LOGIN")
                else ""
            )
        )

    @cached_property
    def exposes(self) -> "Iterable[ExposeLink]":
        return [
            self.expose_other("在线工具集合", "tools", "", "https://tool.lu/"),
            self.expose_other("在线正则表达式", "regex", "", "https://regex101.com/"),
            self.expose_other("正则表达式手册", "regex", "", "https://tool.oschina.net/uploads/apidocs/jquery/regexp.html"),
            self.expose_other("在线json解析", "codeJson", "", "https://www.json.cn/"),
            self.expose_other("DNS查询", "dns", "", "https://tool.chinaz.com/dns/"),
            self.expose_other("图标下载", "progressDownload", "", "https://materialdesignicons.com/"),
        ]

    def on_starting(self):

        categories = {}
        apps = []
        bookmarks = []

        for key, value in vars(ExposeMixin).items():
            if isinstance(value, ExposeCategory):
                categories.setdefault(value, list())

        for container in sorted(self.manager.get_installed_containers(), key=lambda o: o.order):
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
            domain=self.get_config("FLARE_DOAMIN"),
            proxy_url="http://flare:5005",
        )
