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

import rsa
import yaml

from linktools import utils
from linktools.cli import CommandError, subcommand
from linktools.cntr import BaseContainer, ExposeLink, ContainerError, EventContext
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
            AUTHELIA_LDAP_HOST="lldap",
            AUTHELIA_LDAP_PORT=Config.Alias(type=int) | 3890,
            AUTHELIA_LDAP_ADDRESS=Config.Lazy(
                lambda cfg: f"ldap://{cfg.get('AUTHELIA_LDAP_HOST')}:{cfg.get('AUTHELIA_LDAP_PORT')}"),
            AUTHELIA_LDAP_WEB_PORT=Config.Alias(type=int) | 17170,
            AUTHELIA_LDAP_WEB_ADDRESS=Config.Lazy(
                lambda cfg: f"http://{cfg.get('AUTHELIA_LDAP_HOST')}:{cfg.get('AUTHELIA_LDAP_WEB_PORT')}"),
            AUTHELIA_LDAP_USER="admin",
            AUTHELIA_LDAP_PASSWORD=Config.Alias("LLDAP_ADMIN_PASSWORD") | Config.Prompt(cached=True),
            AUTHELIA_LDAP_BASE_DN=Config.Alias("LLDAP_BASE_DN") | "dc=example,dc=org",
            AUTHELIA_MIN_AUTH_LEVEL=Config.Alias(type=int) | 2,
            AUTHELIA_OIDC_CLIENT_SECRET=Config.Alias(cached=True) | utils.random_string(20),
            AUTHELIA_ADMIN_AUTH_ENABLE=Config.Property(type=bool) | True,
        )

    @cached_property
    def exposes(self) -> Iterable[ExposeLink]:
        return [
            self.expose_public("Authelia", "account", "单点登录", self.load_nginx_url(
                "AUTHELIA_DOMAIN", "auth-admin",
                proxy_conf=self.get_source_path("templates", "nginx.conf"),
                auth_enable=self.get_config("AUTHELIA_ADMIN_AUTH_ENABLE"),
                auth_extra={
                    "acl_bypass": ["\\.(css|js)$"],
                    "acl_rule": {
                        "subject": ["group:lldap_admin"],
                    }
                }
            )),
        ]

    @cached_property
    def _key_prefix(self):
        return f"{self.get_config('AUTHELIA_DOMAIN')}_{self.get_config('NGINX_HTTPS_PORT')}"

    @cached_property
    def acl_rules(self):
        result = None

        with self.settings.open() as settings:
            result = settings.get(f"{self._key_prefix}_acl_rules", default=None)
            if result is None:
                result = {}
                settings.set(f"{self._key_prefix}_acl_rules", result)

        return result

    @cached_property
    def oidc_clients(self):
        result = None

        with self.settings.open() as settings:
            result = settings.get(f"{self._key_prefix}_oidc_clients", default=None)
            if result is None:
                port = self.get_config("NGINX_HTTPS_PORT")
                domain = self.get_config("AUTHELIA_DOMAIN")
                auth_url = utils.make_url("https", domain, port)

                client = dict()
                client["ClientID"] = f"{self.manager.project_name}-web-client"
                client["ClientName"] = f"Web Client ({self.manager.project_name})"
                client["ClientSecret"] = self.get_config("AUTHELIA_OIDC_CLIENT_SECRET")
                client["IssuerURL"] = auth_url
                client["AuthorizationURL"] = f"{auth_url}/api/oidc/authorization"
                client["AccessTokenURL"] = f"{auth_url}/api/oidc/token"
                client["ResourceURL"] = f"{auth_url}/api/oidc/userinfo"
                client["RedirectURLs"] = {auth_url}
                client["UserIdentifier"] = "preferred_username"
                client["Scopes"] = "openid profile groups email phone"
                result = [client]
                settings.set(f"{self._key_prefix}_oidc_clients", result)

            client = result[0]
            client["ClientID"] = f"{self.manager.project_name}-web-client"
            client["ClientName"] = f"Web Client ({self.manager.project_name})"
            client["ClientSecret"] = self.get_config("AUTHELIA_OIDC_CLIENT_SECRET")

        return result

    def on_init(self):
        self.start_hooks.append(lambda: self.manager.start_hooks.append(self._update_files))

    def on_check(self, context: EventContext):
        if not self.get_config("NGINX_HTTPS_ENABLE"):
            raise ContainerError("Authelia requires HTTPS. Please set NGINX_HTTPS_ENABLE to true.")

    def _update_files(self):
        secret_path = self.get_app_path("secrets")
        secret_path.mkdir(parents=True, exist_ok=True)
        config_path = self.get_app_path("config")
        config_path.mkdir(parents=True, exist_ok=True)
        template_path = self.get_source_path("templates")

        self.manager.change_file_owner(secret_path, self.manager.user, recursive=True)
        self.manager.change_file_mode(secret_path, 0o700, recursive=True)
        self.manager.change_file_owner(config_path, self.manager.user, recursive=True)
        self.manager.change_file_mode(config_path, 0o700, recursive=True)

        self._create_secret_file(secret_path / "jwt_secret")
        self._create_secret_file(secret_path / "session_secret")
        self._create_secret_file(secret_path / "storage_encryption_key")
        self._create_secret_file(secret_path / "oidc_hmac_secret")
        self._create_pem_file(secret_path / "identity_providers_oidc_jwks")
        utils.write_file(secret_path / "authentication_backend_ldap_password", self.get_config("LLDAP_ADMIN_PASSWORD"))

        self.render_template(template_path / "configuration.yml", config_path / "configuration.yml")
        self.render_template(template_path / "configuration.acl.yml", config_path / "configuration.acl.yml")
        self.render_template(template_path / "configuration.2fa.yml", config_path / "configuration.2fa.yml")
        self.render_template(template_path / "configuration.oidc.yml", config_path / "configuration.oidc.yml")

        self.manager.change_file_owner(secret_path, "root", recursive=True)
        self.manager.change_file_owner(config_path, "root", recursive=True)

        with self.settings.open() as settings:
            settings.set(f"{self._key_prefix}_acl_rules", self.acl_rules)
            settings.set(f"{self._key_prefix}_oidc_clients", self.oidc_clients)

    def on_stopped(self, context: EventContext):
        if context.is_full_containers:
            self.on_removed(context)

    def on_removed(self, context: EventContext):
        with self.settings.open() as settings:
            settings.pop(f"{self._key_prefix}_acl_rules", None)
            settings.pop(f"{self._key_prefix}_oidc_clients", None)

    @subcommand("show-notification", help="show notification")
    def on_show_notification(self):
        path = self.get_app_path("config", "notification.txt")
        if path.exists():
            self.logger.info(utils.read_file(path, text=True))
        else:
            self.logger.warning("No notification.")

    @subcommand("list-oidc-clients", help="list OIDC clients")
    def on_list_oidc_clients(self):
        self.logger.info(
            yaml.dump(self.oidc_clients, sort_keys=False)
        )

    @subcommand("list-acl-rules", help="list acl rules")
    def on_list_acl_rules(self):
        self.logger.info(
            yaml.dump(self.acl_rules, sort_keys=False)
        )

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

        public_key, private_key = rsa.newkeys(nbits=2048, exponent=65537)
        private_pem = private_key.save_pkcs1(format="PEM")
        utils.write_file(path, private_pem)
