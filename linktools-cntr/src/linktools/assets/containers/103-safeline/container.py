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

from linktools.cli import subcommand
from linktools.cntr import BaseContainer, ExposeLink
from linktools.core import Config
from linktools.decorator import cached_property


class Container(BaseContainer):

    @property
    def dependencies(self) -> "Iterable[str]":
        return ["nginx"]

    @cached_property
    def configs(self):
        return dict(
            SAFELINE_TAG="latest",
            SAFELINE_IMAGE_PREFIX="chaitin",
            SAFELINE_DOMAIN=self.get_nginx_domain(),
            SAFELINE_POSTGRES_PASSWORD="Pg-pAssw0rd",
            SAFELINE_SUBNET_PREFIX="172.22.242",
            SAFELINE_ARCH_SUFFIX="",
            SAFELINE_RELEASE="",
            SAFELINE_PORT=Config.Property(type=int) | 9200,
        )

    @cached_property
    def exposes(self) -> Iterable[ExposeLink]:
        return [
            self.expose_public("Safeline", "alienOutline", "雷池WAF", self.load_nginx_url(
                "SAFELINE_DOMAIN",
                proxy_url="https://safeline-mgt:1443",
                auth_enable=True,
            )),
            self.expose_container("Safeline", "alienOutline", "雷池WAF", self.load_port_url(
                "SAFELINE_PORT",
                https=True
            )),
        ]

    @subcommand("reset-admin", help="reset safeline admin password")
    def on_reset_admin(self):
        self.manager.create_docker_process(
            "exec", "-it", self.get_service_name("safeline-mgt"),
            "resetadmin"
        ).call()
