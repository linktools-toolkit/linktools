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
import json
import os
import shutil
from typing import Dict, Any

from linktools import utils
from linktools.cntr import BaseContainer, ContainerError, EventContext
from linktools.core import Config
from linktools.decorator import cached_property
from linktools.metadata import __missing__
from linktools.types import PathType


class Container(BaseContainer):

    @cached_property
    def dnsapi(self):
        with open(self.get_source_path("dnsapi.json"), "rt") as fd:
            return json.load(fd)

    @cached_property
    def configs(self):
        return dict(
            NGINX_TAG="stable-alpine",
            NGINX_WILDCARD_DOMAIN=Config.Alias("WILDCARD_DOMAIN") | False,
            NGINX_ROOT_DOMAIN=Config.Alias("ROOT_DOMAIN") | Config.Prompt(cached=True) | "_",
            NGINX_HTTP_PORT=Config.Alias("HTTP_PORT", type=int) | Config.Prompt(cached=True) | 80,
            NGINX_HTTPS_ENABLE=Config.Alias("HTTPS_ENABLE", type=bool) | Config.Confirm(cached=True) | True,
            NGINX_HTTPS_PORT=Config.Alias("HTTPS_PORT", type=int) | Config.Lazy(
                lambda cfg:
                Config.Prompt(type=int, cached=True) | 443
                if cfg.get("NGINX_HTTPS_ENABLE")
                else Config.Alias(type=int) | 0
            ),
            NGINX_INDEX_URL=Config.Lazy(
                lambda cfg: self._get_default_index_url()
            ),
            NGINX_WAF_ENABLE=Config.Alias("WAF_ENABLE", type=bool) | Config.Lazy(
                lambda cfg: self.manager.containers["safeline"].enable
            ),
            NGINX_WAF_PORT=Config.Alias("WAF_PORT", type=int) | Config.Lazy(
                lambda cfg:
                Config.Prompt(type=int, cached=True) | 8000
                if cfg.get("NGINX_WAF_ENABLE")
                else Config.Alias(type=int) | 0
            ),
            NGINX_AUTH_ENABLE=Config.Alias("AUTH_ENABLE", type=bool) | Config.Lazy(
                lambda cfg: self.manager.containers["authelia"].enable
            ),
            ACME_DNS_API=Config.Lazy(
                lambda cfg:
                Config.Prompt(choices=self.dnsapi.keys(), cached=True)
                if cfg.get("NGINX_HTTPS_ENABLE")
                else Config.Property(type=str) | ""
            )
        )

    @cached_property
    def extend_configs(self):
        configs = {}
        if self.get_config("NGINX_HTTPS_ENABLE"):
            dns_api = self.get_config("ACME_DNS_API")
            if dns_api not in self.dnsapi:
                raise ContainerError(f"Not supported dns_api: {dns_api}")
            env_vars = self.dnsapi.get(dns_api).get("env", {})
            for env_var, meta in env_vars.items():
                configs[env_var] = Config.Prompt(cached=True, allow_empty=meta.get("required", True))
        return configs

    def _get_default_index_url(self):
        host = "www.google.com" \
            if self.get_config("NGINX_ROOT_DOMAIN") in ("", "_", "localhost") \
            else self.get_config("NGINX_ROOT_DOMAIN")
        if self.get_config("NGINX_HTTPS_ENABLE"):
            scheme = "https"
            port = self.get_config("NGINX_HTTPS_PORT")
        else:
            scheme = "http"
            port = self.get_config("NGINX_HTTP_PORT")
        return utils.make_url(scheme, host, port)

    def on_init(self):
        self.start_hooks.append(lambda: self.manager.start_hooks.append(self._update_files))

    def on_check(self, context: EventContext):
        if self.get_config("NGINX_WILDCARD_DOMAIN") and self.get_config("NGINX_ROOT_DOMAIN") in ("", "_", "localhost"):
            raise ContainerError("Wildcard domain is enabled but root domain is not set.")
        if self.get_config("NGINX_WAF_ENABLE") and not self.manager.containers["safeline"].enable:
            raise ContainerError("NGINX_WAF_ENABLE is true but safeline container is not enabled.")
        if self.get_config("NGINX_AUTH_ENABLE") and not self.manager.containers["authelia"].enable:
            raise ContainerError("NGINX_AUTH_ENABLE is true but authelia container is not enabled.")

    def _update_files(self):
        utils.clear_directory(self.get_app_path("conf.d"))

        # 初始化snippets
        snippets_path = self.get_app_path("conf.d", "snippets")
        snippets_path.mkdir(parents=True, exist_ok=True)

        waf_enable = self.get_config("NGINX_WAF_ENABLE")
        auth_enable = self.get_config("NGINX_AUTH_ENABLE")
        self.render_template(
            self.get_source_path("templates", "header.conf"),
            self.get_app_path("conf.d", "snippets", "header.conf"),
            X_HEADER_ENABLE=not waf_enable
        )
        if waf_enable:
            self.render_template(
                self.get_source_path("templates", "waf.conf"),
                self.get_app_path("conf.d", "snippets", "waf.conf"),
            )
        if auth_enable:
            self.render_template(
                self.get_source_path("templates", "auth.conf"),
                self.get_app_path("conf.d", "snippets", "auth.conf"),
            )

        # 初始化conf.d
        for container in self.manager.get_installed_containers():
            path = self.get_app_path("temporary", container.name)
            if os.path.isdir(path):
                shutil.copytree(
                    path,
                    self.get_app_path("conf.d", create_parent=True),
                    dirs_exist_ok=True,
                )
        if not self.get_app_path("conf.d", "_.conf").exists():
            self.write_conf(
                self, "_",
                proxy_name="default",
                proxy_conf=self.get_source_path("templates", "index.conf"),
                flush=True,
            )

    def on_started(self, context: EventContext):
        # 更新证书（如果启用HTTPS）
        if self.get_config("NGINX_HTTPS_ENABLE"):
            root_domain = self.get_config("NGINX_ROOT_DOMAIN")
            dns_api = self.get_config("ACME_DNS_API")
            self.logger.info("Renew nginx certificates if necessary.")
            self.manager.create_docker_process(
                "exec", "-it", self.get_service_name("nginx"),
                "sh", "-c", f"acme.sh --renew --issue "
                            f"--domain {root_domain} --domain *.{root_domain} "
                            f"--dns {dns_api} "
                            f"1>/dev/null"
            ).call()
            self.manager.create_docker_process(
                "exec", "-it", self.get_service_name("nginx"),
                "sh", "-c", f"acme.sh --install-cert "
                            f"--domain {root_domain} --domain *.{root_domain} "
                            f"--cert-file /etc/certs/{root_domain}_cert.pem "
                            f"--key-file /etc/certs/{root_domain}_key.pem "
                            f"--fullchain-file /etc/certs/{root_domain}_fullchain.pem "
                            f"1>/dev/null"
            ).call()

        # 重启nginx
        self.manager.create_docker_process(
            "exec", "-it", self.get_service_name("nginx"),
            "sh", "-c", "killall nginx 1>/dev/null 2>&1"
        ).call()

    def on_stopped(self, context: EventContext):
        if context.is_full_containers:
            self.on_removed(context)
            return
        for container in context.target_containers:
            path = self.get_app_path("temporary", container.name)
            if path.exists():
                utils.remove_file(path)

    def on_removed(self, context: EventContext):
        utils.clear_directory(self.get_app_path("temporary"))
        utils.clear_directory(self.get_app_path("conf.d"))

    def write_conf(
        self, container: BaseContainer, domain: str, *,
        proxy_name: str = __missing__, proxy_conf: PathType = __missing__, proxy_url: str = __missing__,
        https_enable: bool = __missing__, waf_enable: bool = __missing__,
        auth_enable: bool = False, auth_extra: "Dict[str, Any]" = __missing__,
        flush: bool = False,
    ):

        if flush:
            conf_path = self.get_app_path("conf.d", f"{domain}.conf")
            sub_conf_path = self.get_app_path("conf.d", f"{domain}_confs", f"{proxy_name or container.name}.conf")
        else:
            conf_path = self.get_app_path("temporary", container.name, f"{domain}.conf")
            sub_conf_path = self.get_app_path("temporary", container.name, f"{domain}_confs", f"{proxy_name or container.name}.conf")

        try:
            if not domain:
                raise ContainerError("not found domain")
            if not proxy_conf:
                if not proxy_url:
                    raise ContainerError("not found url")
                proxy_conf = self.get_source_path("templates", "default.conf")

            if https_enable is __missing__:
                https_enable = True
            if waf_enable is __missing__:
                waf_enable = True
            https_enable = https_enable and self.get_config("NGINX_HTTPS_ENABLE")
            waf_enable = waf_enable and self.get_config("NGINX_WAF_ENABLE")

            if auth_enable:
                if not self.get_config("NGINX_AUTH_ENABLE", type=bool):
                    self.logger.warning(f"NGINX_AUTH_ENABLE is false, disable auth in {container}")
                    auth_enable = False

            context = dict(
                DOMAIN=domain,
                HTTPS_ENABLE=https_enable,
                WAF_ENABLE=waf_enable,
                AUTH_ENABLE=auth_enable,
                AUTH_HEADERS=auth_extra.get("auth_headers", None) if auth_extra else None,
                AUTH_BYPASS=auth_extra.get("acl_bypass", None) if auth_extra else None,
            )

            conf_path.parent.mkdir(parents=True, exist_ok=True)
            sub_conf_path.parent.mkdir(parents=True, exist_ok=True)
            container.render_template(
                self.get_source_path("templates", "server.conf"),
                conf_path,
                **context,
            )
            if proxy_conf is not __missing__ or proxy_url is not __missing__:
                container.render_template(
                    proxy_conf,
                    sub_conf_path,
                    PROXY_URL=proxy_url,
                    **context,
                )
            if auth_enable:
                authelia = self.manager.containers["authelia"]
                authelia.write_nginx_conf(
                    domain=domain,
                    proxy_name="auth_location",
                    proxy_conf=self.get_source_path("templates", "auth_location.conf"),
                )
                if auth_extra:
                    uris = auth_extra.get("oidc_redirect_uris", None)
                    if uris:
                        oidc_redirect_uris = authelia.oidc_clients[0].get("RedirectURLs")
                        for uri in uris:
                            if not uri:
                                self.logger.info(f"{container} invalid oidc redirect uri: None, skip.")
                                continue
                            scheme = "https" if https_enable else "http"
                            port = self.get_config("NGINX_HTTPS_PORT" if https_enable else "NGINX_HTTP_PORT")
                            base_url = utils.make_url(scheme, domain, port)
                            redirect_uri = uri.format(scheme=scheme, domain=domain, port=port, base_url=base_url)
                            if not redirect_uri:
                                self.logger.info(f"{container} invalid oidc redirect uri: {uri}, skip.")
                                continue
                            oidc_redirect_uris.add(redirect_uri)

                    acl_rule = auth_extra.get("acl_rule", None)
                    if acl_rule:
                        target_acl_rule = authelia.acl_rules.setdefault(domain, {})
                        target_acl_rule["Subject"] = acl_rule.get("subject", None)
                        target_acl_rule["Policy"] = acl_rule.get("policy", None)

        except ContainerError as e:
            self.logger.debug(f"{container} write nginx conf: {e}, skip.")

            utils.remove_file(sub_conf_path)
            if sub_conf_path.parent.exists():
                try:
                    if not any(f.endswith(".conf") for f in os.listdir(sub_conf_path.parent)):
                        utils.remove_file(sub_conf_path.parent)
                        utils.remove_file(conf_path)
                except:
                    pass
