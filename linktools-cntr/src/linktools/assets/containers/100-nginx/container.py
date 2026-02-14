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
import re
import shutil
import textwrap

from linktools import utils
from linktools.core import Config
from linktools.decorator import cached_property
from linktools.cntr import BaseContainer


class Container(BaseContainer):

    @cached_property
    def keys(self):
        # dnsapi.txt 内容从 https://github.com/acmesh-official/acme.sh/wiki/dnsapi 拷贝
        path = os.path.join(os.path.dirname(__file__), "dnsapi.txt")
        data = utils.read_file(path, text=True)
        pattern = re.compile(r'export +(\w+)="?')
        return sorted(list(set(pattern.findall(data))))

    @cached_property
    def configs(self):
        return dict(
            NGINX_TAG="1.29.1-alpine",
            NGINX_WILDCARD_DOMAIN=Config.Alias("WILDCARD_DOMAIN") | False,
            NGINX_ROOT_DOMAIN=Config.Alias("ROOT_DOMAIN") | Config.Prompt(cached=True) | "_",
            NGINX_HTTP_PORT=Config.Alias("HTTP_PORT", type=int) | Config.Prompt(cached=True) | 80,
            NGINX_HTTPS_ENABLE=Config.Alias("HTTPS_ENABLE") | Config.Confirm(cached=True) | True,
            NGINX_HTTPS_PORT=Config.Alias("HTTPS_PORT") | Config.Lazy(
                lambda cfg:
                Config.Prompt(type=int, cached=True) | 443
                if cfg.get("NGINX_HTTPS_ENABLE")
                else Config.Alias(type=int) | 0
            ),
            NGINX_INDEX_URL="https://www.google.com/",
            NGINX_WAF_ENABLE=Config.Lazy(
                lambda cfg: self.manager.containers["safeline"].enable
            ),
            NGINX_WAF_PORT=Config.Lazy(
                lambda cfg:
                Config.Prompt(type=int, cached=True) | 8000
                if cfg.get("NGINX_WAF_ENABLE")
                else Config.Alias(type=int) | 0
            ),
            ACME_DNS_API=Config.Lazy(
                lambda cfg:
                Config.Error(textwrap.dedent(
                    """
                    Ensure ACME_DNS_API config matches --dns parameter in acme command is set.
                    · Also, set corresponding environment variables.
                    · For details, see: https://github.com/acmesh-official/acme.sh/wiki/dnsapi.
                    · Example command:
                      $ ct-cntr config set ACME_DNS_API=dns_ali Ali_Key=xxx Ali_Secret=yyy
                    """
                ))
                if cfg.get("NGINX_HTTPS_ENABLE")
                else Config.Property(type=str) | ""
            )
        )

    def on_started(self):
        utils.clear_directory(self.get_app_path("conf.d"))

        # 初始化snippets
        snippets_path = self.get_app_path("conf.d", "snippets")
        snippets_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            self.get_source_path("snippets", "header.conf"),
            self.get_app_path("conf.d", "snippets", "header.conf"),
        )
        shutil.copy2(
            self.get_source_path("snippets", "x-header.conf"),
            self.get_app_path("conf.d", "snippets", "x-header.conf"),
        )
        waf_enable = self.get_config("NGINX_WAF_ENABLE")
        if waf_enable:
            self.render_template(
                self.get_source_path("snippets", "default.conf"),
                self.get_app_path("conf.d", "snippets", "waf.conf"),
                PROXY_URL=f"http://safeline-tengine:{self.get_config('NGINX_WAF_PORT')}",
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
            self.write_nginx_conf(
                "_",
                proxy_name="default",
                proxy_conf=self.get_source_path("snippets", "index.conf"),
                proxy_url=self.get_config("NGINX_INDEX_URL"),
            )
            path = self.get_app_path("temporary", self.name)
            if os.path.isdir(path):
                shutil.copytree(
                    path,
                    self.get_app_path("conf.d", create_parent=True),
                    dirs_exist_ok=True,
                )

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

    def on_stopped(self):
        utils.clear_directory(self.get_app_path("temporary"))
        utils.clear_directory(self.get_app_path("conf.d"))
