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
import os
from typing import Iterable

import yaml

from linktools import utils
from linktools.cli import CommandError, subcommand
from linktools.cntr import BaseContainer, ExposeLink
from linktools.core import Config
from linktools.decorator import cached_property


class Container(BaseContainer):

    @property
    def dependencies(self) -> "Iterable[str]":
        return ["nginx", "lldap"]

    @cached_property
    def configs(self):
        return dict(
            AUTHELIA_TAG="latest",
            AUTHELIA_DOMAIN=self.get_nginx_domain("sso"),
        )

    @cached_property
    def exposes(self) -> Iterable[ExposeLink]:
        return [
            self.expose_public("Authelia", "account", "单点登录", self.load_nginx_url(
                "AUTHELIA_DOMAIN", "auth-admin",
                proxy_conf=self.get_source_path("templates", "nginx.conf"),
            )),
        ]

    def on_starting(self):
        # TODO: 校验环境
        template_path = self.get_source_path("templates")

        secret_path = self.get_app_path("secrets")
        secret_path.mkdir(parents=True, exist_ok=True)
        self._create_secret_file(secret_path / "jwt_secret")
        self._create_secret_file(secret_path / "session_secret")
        self._create_secret_file(secret_path / "storage_encryption_key")
        self._create_secret_file(secret_path / "oidc_hmac_secret")
        utils.write_file(secret_path / "authentication_backend_ldap_password", self.get_config("LLDAP_ADMIN_PASSWORD"))

        config_path = self.get_app_path("config")
        config_path.mkdir(parents=True, exist_ok=True)
        self.render_template(template_path / "configuration.yml", config_path / "configuration.yml")
        self.render_template(template_path / "configuration.acl.yml", config_path / "configuration.acl.yml")
        self.render_template(template_path / "configuration.2fa.yml", config_path / "configuration.2fa.yml")

    @classmethod
    def _create_secret_file(cls, path, length=48):
        if os.path.exists(path):
            if os.path.isfile(path):
                return
            raise CommandError(f"Path {path} exists and is not a file.")

        utils.write_file(path, utils.random_string(length))

    @subcommand("notify", help="show notification")
    def on_notify(self):
        path = self.get_app_path("config", "notification.txt")
        if path.exists():
            self.logger.info(utils.read_file(path))
        else:
            self.logger.warning("No notification.")
