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

from linktools.core import Config
from linktools.decorator import cached_property
from linktools.cntr import BaseContainer, ExposeLink


class Container(BaseContainer):

    @cached_property
    def configs(self):
        return dict(
            SAFELINE_TAG="latest",
            SAFELINE_IMAGE_PREFIX="chaitin",
            SAFELINE_POSTGRES_PASSWORD="Pg-pAssw0rd",
            SAFELINE_SUBNET_PREFIX="172.22.242",
            SAFELINE_ARCH_SUFFIX="",
            SAFELINE_RELEASE="",
            SAFELINE_EXPOSE_PORT=Config.Property(type=int) | 9443,
        )

    @cached_property
    def exposes(self) -> Iterable[ExposeLink]:
        return [
            self.expose_container("Safeline", "wallFire", "雷池", self.load_port_url(
                "SAFELINE_EXPOSE_PORT", https=True
            )),
        ]
