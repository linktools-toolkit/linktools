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

from linktools import utils
from linktools.cli import CommandError
from linktools.cntr import BaseContainer, ExposeLink, ContainerError
from linktools.core import Config
from linktools.decorator import cached_property


class Container(BaseContainer):

    @property
    def dependencies(self) -> "Iterable[str]":
        return ["nginx"]

    @cached_property
    def configs(self):
        def get_base_dn(cfg):
            domain = cfg.get("NGINX_ROOT_DOMAIN")
            parts = domain.split(".")
            return ",".join([f"dc={part}" for part in parts])

        return dict(
            LLDAP_TAG="stable",
            LLDAP_DOMAIN=self.get_nginx_domain("ldap"),
            LLDAP_PORT=Config.Alias(type=int) | 0,
            LLDAP_WEB_PORT=Config.Alias(type=int) | 17170,
            LLDAP_BASE_DN=Config.Lazy(lambda cfg: get_base_dn(cfg)),
            LLDAP_ADMIN_PASSWORD=Config.Prompt(cached=True) | utils.random_string(20),
        )

    @cached_property
    def exposes(self) -> Iterable[ExposeLink]:
        return [
            self.expose_container("LDAP", "account", "账号管理", self.load_port_url(
                "LLDAP_WEB_PORT",
                https=False,
            )),
        ]

    def on_check(self):
        domain = self.get_config("NGINX_ROOT_DOMAIN")
        if not domain or "." not in domain:
            raise ContainerError(f"Invalid domain `{domain}` for LDAP, "
                                 f"Please set NGINX_ROOT_DOMAIN to a valid domain (e.g., example.com).")

    def on_starting(self):
        secret_path = self.get_app_path("secrets")
        secret_path.mkdir(parents=True, exist_ok=True)

        data_path = self.get_app_path("data")
        data_path.mkdir(parents=True, exist_ok=True)

        template_path = self.get_source_path("templates")

        self.manager.change_file_owner(secret_path, self.manager.user, recursive=True)
        self.manager.change_file_mode(secret_path, 0o700, recursive=True)
        self.manager.change_file_owner(data_path, self.manager.user, recursive=True)
        self.manager.change_file_mode(data_path, 0o700, recursive=True)

        self._create_secret_file(secret_path / "jwt_secret", length=64)
        utils.write_file(secret_path / "ldap_user_pass", self.get_config("LLDAP_ADMIN_PASSWORD"))
        self.render_template(template_path / "lldap_config.toml", data_path / "lldap_config.toml")

        self.manager.change_file_owner(secret_path, "root", recursive=True)
        self.manager.change_file_owner(data_path, "root", recursive=True)

    @classmethod
    def _create_secret_file(cls, path, length=48):
        if os.path.exists(path):
            if not os.path.isfile(path):
                raise CommandError(f"Path {path} exists and is not a file.")
            return

        utils.write_file(path, utils.random_string(length))
