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
from linktools.cli import CommandError, subcommand
from linktools.cntr import BaseContainer, ExposeLink, ContainerError
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
            AUTHELIA_MIN_AUTH_LEVEL=Config.Alias(type=int) | 2,
            AUTHELIA_OIDC_CLIENT_SECRET=Config.Alias(type=str, default=utils.random_string(20)),
        )

    @cached_property
    def exposes(self) -> Iterable[ExposeLink]:
        return [
            self.expose_public("Authelia", "account", "单点登录", self.load_nginx_url(
                "AUTHELIA_DOMAIN", "auth-admin",
                proxy_conf=self.get_source_path("templates", "nginx.conf"),
            )),
        ]

    @cached_property
    def data(self):
        return {
            "oidc_redirect_uris": set()
        }

    def on_init(self):
        self.start_hooks.append(self._update_files)

    def on_check(self):
        if not self.get_config("NGINX_HTTPS_ENABLE", type=bool):
            raise ContainerError("Authelia requires HTTPS. Please set NGINX_HTTPS_ENABLE to true.")

    def _update_files(self):
        template_path = self.get_source_path("templates")

        secret_path = self.get_app_path("secrets")
        secret_path.mkdir(parents=True, exist_ok=True)
        self._create_secret_file(secret_path / "jwt_secret")
        self._create_secret_file(secret_path / "session_secret")
        self._create_secret_file(secret_path / "storage_encryption_key")
        self._create_secret_file(secret_path / "oidc_hmac_secret")
        self._create_pem_file(secret_path / "identity_providers_oidc_jwks")
        utils.write_file(secret_path / "authentication_backend_ldap_password", self.get_config("LLDAP_ADMIN_PASSWORD"))

        config_path = self.get_app_path("config")
        config_path.mkdir(parents=True, exist_ok=True)
        self.render_template(template_path / "configuration.yml", config_path / "configuration.yml")
        self.render_template(template_path / "configuration.acl.yml", config_path / "configuration.acl.yml")
        self.render_template(template_path / "configuration.2fa.yml", config_path / "configuration.2fa.yml")
        self.render_template(template_path / "configuration.oidc.yml", config_path / "configuration.oidc.yml")

    @subcommand("show-notification", help="show notification")
    def on_show_notification(self):
        path = self.get_app_path("config", "notification.txt")
        if path.exists():
            self.logger.info(utils.read_file(path))
        else:
            self.logger.warning("No notification.")

    @classmethod
    def _create_secret_file(cls, path, length=48):
        if os.path.exists(path):
            if not os.path.isfile(path):
                raise CommandError(f"Path {path} exists and is not a file.")
            return

        utils.write_file(path, utils.random_string(length))

    @classmethod
    def _create_pem_file(cls, path):
        if os.path.exists(path):
            if not os.path.isfile(path):
                raise CommandError(f"Path {path} exists and is not a file.")
            return

        from cryptography.hazmat.primitives.asymmetric import rsa
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )

        # 转换为 PEM 格式（与 openssl 默认一致）
        from cryptography.hazmat.primitives import serialization
        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,  # 等价 openssl genrsa
            encryption_algorithm=serialization.NoEncryption()
        )

        utils.write_file(path, pem)
