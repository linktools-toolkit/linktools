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

from linktools import utils
from linktools.cntr import BaseContainer, ExposeLink
from linktools.core import Config
from linktools.decorator import cached_property


class Container(BaseContainer):

    @cached_property
    def configs(self):
        return dict(
            AUTHENTIK_TAG="2025.12.4",
            AUTHENTIK_IMAGE="ghcr.io/goauthentik/server",
            AUTHENTIK_DOMAIN=self.get_nginx_domain(),
            AUTHENTIK_EXPOSE_PORT=Config.Property(type=int) | 9100,
            AUTHENTIK_SECRET_KEY=Config.Alias(default=utils.random_string(36), cached=True),
            AUTHENTIK_POSTGRES_PASSWORD=Config.Alias(default=utils.random_string(36), cached=True),
        )

    @cached_property
    def exposes(self) -> Iterable[ExposeLink]:
        return [
            self.expose_public("Authentik", "account", "身份认证", self.load_nginx_url(
                "AUTHENTIK_DOMAIN",
                proxy_url="http://authentik-server:9000"
            )),
            self.expose_container("Authentik", "account", "身份认证", self.load_port_url(
                "AUTHENTIK_EXPOSE_PORT", https=True
            )),
        ]
