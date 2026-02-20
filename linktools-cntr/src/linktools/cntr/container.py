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
import textwrap
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Dict, Any, List, Optional, Callable, Iterable

import yaml
from jinja2 import Environment, TemplateError, FileSystemLoader
from linktools import utils
from linktools.cli import subcommand, subcommand_argument
from linktools.core import Config
from linktools.decorator import cached_property
from linktools.metadata import __missing__
from linktools.rich import choose, confirm
from linktools.types import PathType, Error, FileCache

if TYPE_CHECKING:
    from linktools.types import T, ConfigType
    from .manager import ContainerManager


class ExposeCategory:

    def __init__(self, name: str, desc: str):
        self.name = name
        self.desc = desc

    def __call__(self, name: str, icon: str, desc: str, url: str):
        return ExposeLink(self, name, icon, desc or name, url)


class ExposeLink:

    def __init__(self, category: ExposeCategory, name: str, icon: str, desc: str, url: str):
        self.category = category
        self.name = name
        self.icon = icon
        self.desc = desc
        self.url = url

    @property
    def is_valid(self) -> bool:
        return not not self.url


class ExposeMixin:
    expose_public = ExposeCategory("public", "Public")
    expose_private = ExposeCategory("private", "Private")
    expose_container = ExposeCategory("container", "Container")
    expose_other = ExposeCategory("other", "Other")

    def load_config_url(self: "BaseContainer", key: str, *path: str):
        url = self.get_config(key, type=str, default=None)
        if url:
            return utils.make_url(url, *path)
        return None

    def load_port_url(self: "BaseContainer", key: str, *path: str, https: bool = True):
        port = self.get_config(key, type=int, default=0)
        if 0 < port < 65535:
            return utils.make_url(f"{'https' if https else 'http'}://{self.manager.host}:{port}", *path)
        return None

    def load_nginx_url(
            self: "BaseContainer", key: str, *path: str,
            proxy_name: str = __missing__, proxy_conf: PathType = __missing__, proxy_url: str = __missing__,
            https_enable: bool = __missing__, waf_enable: bool = __missing__, auth_enable: bool = False,
    ):
        domain = self.get_config(key, type=str, default=None)
        if domain:
            if https_enable is __missing__:
                https_enable = True
            if waf_enable is __missing__:
                waf_enable = True
            https_enable = https_enable and self.get_config("NGINX_HTTPS_ENABLE", type=bool)
            waf_enable = waf_enable and self.get_config("NGINX_WAF_ENABLE", type=bool)
            if proxy_conf or proxy_url:
                self.start_hooks.append(lambda: self.write_nginx_conf(
                    domain=domain,
                    proxy_name=proxy_name,
                    proxy_conf=proxy_conf,
                    proxy_url=proxy_url,
                    https_enable=https_enable,
                    waf_enable=waf_enable,
                    auth_enable=auth_enable,
                ))
            scheme = "https" if https_enable else "http"
            port = self.get_config("NGINX_HTTPS_PORT" if https_enable else "NGINX_HTTP_PORT", type=int)
            return utils.make_url(f"{scheme}://{domain}:{port}/", *path)
        return None


class NginxMixin:

    def get_nginx_domain(self: "BaseContainer", name: str = None):

        def get_domain(cfg: Config):
            if not self.manager.containers["nginx"].enable:
                return ""
            if not cfg.get("NGINX_WILDCARD_DOMAIN", type=bool):
                return cfg.get("NGINX_ROOT_DOMAIN")
            root_domain = cfg.get("NGINX_ROOT_DOMAIN")
            if root_domain in ("_", "0.0.0.0"):
                return root_domain
            if name is None:
                return f"{self.name}.{root_domain}"
            elif name.strip() == "":
                return root_domain
            return f"{name}.{root_domain}"

        return Config.Lazy(get_domain)

    def write_nginx_conf(
            self: "BaseContainer", domain: str, *,
            proxy_name: str = __missing__, proxy_conf: PathType = __missing__, proxy_url: str = __missing__,
            https_enable: bool = __missing__, waf_enable: bool = __missing__, auth_enable: bool = False,
            flush: bool = False,
    ):

        nginx = self.manager.containers["nginx"]

        if flush:
            conf_path = nginx.get_app_path("conf.d", f"{domain}.conf")
            sub_conf_path = nginx.get_app_path("conf.d", f"{domain}_confs", f"{proxy_name or self.name}.conf")
        else:
            conf_path = nginx.get_app_path("temporary", self.name, f"{domain}.conf")
            sub_conf_path = nginx.get_app_path("temporary", self.name, f"{domain}_confs", f"{proxy_name or self.name}.conf")

        try:
            if not nginx.enable:
                raise ContainerError("nginx is disable")
            if not domain:
                raise ContainerError("not found domain")
            if not proxy_conf:
                if not proxy_url:
                    raise ContainerError("not found url")
                proxy_conf = nginx.get_source_path("templates", "default.conf")

            if https_enable is __missing__:
                https_enable = True
            if waf_enable is __missing__:
                waf_enable = True
            https_enable = https_enable and self.get_config("NGINX_HTTPS_ENABLE", type=bool)
            waf_enable = waf_enable and self.get_config("NGINX_WAF_ENABLE", type=bool)

            if auth_enable:
                if not nginx.get_config("NGINX_AUTH_ENABLE", type=bool):
                    self.logger.warning("Authelia server is disable")
                    auth_enable = False

            context = dict(
                DOMAIN=domain,
                HTTPS_ENABLE=https_enable,
                WAF_ENABLE=waf_enable,
                AUTH_ENABLE=auth_enable,
            )

            conf_path.parent.mkdir(parents=True, exist_ok=True)
            sub_conf_path.parent.mkdir(parents=True, exist_ok=True)
            self.render_template(
                nginx.get_source_path("templates", "server.conf"),
                conf_path,
                **context,
            )
            if proxy_conf is not __missing__ or proxy_url is not __missing__:
                self.render_template(
                    proxy_conf,
                    sub_conf_path,
                    PROXY_URL=proxy_url,
                    **context,
                )
            if auth_enable:
                self.manager.containers["authelia"].write_nginx_conf(
                    domain=domain,
                    proxy_conf=nginx.get_source_path("templates", "auth_location.conf"),
                )

        except ContainerError as e:
            self.logger.debug(f"{self} write nginx conf: {e}, skip.")

            utils.remove_file(sub_conf_path)
            if sub_conf_path.parent.exists():
                try:
                    if not any(f.endswith(".conf") for f in os.listdir(sub_conf_path.parent)):
                        utils.remove_file(sub_conf_path.parent)
                        utils.remove_file(conf_path)
                except:
                    pass


class ContainerError(Error):
    pass


class ContainerTemplateError(ContainerError):
    pass


class AbstractMetaClass(type):

    def __new__(mcs, name, bases, namespace):
        if "__abstract__" not in namespace:
            namespace["__abstract__"] = False
        return super().__new__(mcs, name, bases, namespace)


class BaseContainer(ExposeMixin, NginxMixin, metaclass=AbstractMetaClass):
    __abstract__ = True

    def __init__(self, manager: "ContainerManager", root_path: PathType, name: str = None):
        name = name or self.__module__
        index = name.rfind(".")
        if index >= 0:
            name = name[index + 1:]
        match = re.match(r"^(\d{1,3})-(.*)$", name, re.M | re.I)
        if match:
            self._order = int(match.group(1))
            self._name = match.group(2)
        else:
            self._order = 900
            self._name = name
        self._enable = False
        self.manager = manager
        self.logger = manager.logger
        self.root_path = root_path

    @property
    def name(self) -> str:
        return self._name

    @cached_property
    def description(self) -> str:
        return textwrap.dedent((self.__doc__ or "").strip())

    @property
    def order(self) -> int:
        return self._order

    @property
    def enable(self) -> bool:
        return self._enable

    @enable.setter
    def enable(self, value: bool):
        self._enable = value

    @property
    def dependencies(self) -> "Iterable[str]":
        return []

    @property
    def configs(self) -> Dict[str, Any]:
        return {}

    @property
    def exposes(self) -> "Iterable[ExposeLink]":
        return []

    @property
    def settings(self) -> FileCache:
        return FileCache(self.manager.setting_path / "app" / self.name)

    @cached_property
    def docker_compose(self) -> Optional[Dict[str, Any]]:
        with self.settings.open() as settings:
            mount_paths = settings.get("mount_paths", {})
            for name in self.manager.docker_compose_names:
                path = self.get_source_path(name)
                if not os.path.exists(path):
                    continue
                data = self.render_template(path)
                data = yaml.safe_load(data)
                if "services" in data and isinstance(data["services"], dict):
                    for name, service in data["services"].items():
                        if not isinstance(service, dict):
                            continue
                        service.setdefault("container_name", f"{self.manager.project_name}-{name}")
                        service.setdefault("hostname", name)
                        service.setdefault("restart", self.get_config("SERVICE_RESTART_POLICY"))
                        service.setdefault("logging", {
                            "driver": self.get_config("SERVICE_LOG_DRIVER"),
                            "options": {
                                "max-size": self.get_config("SERVICE_LOG_MAX_SIZE"),
                            }
                        })
                        if "image" not in service and "build" not in service:
                            path = self.get_docker_file_path()
                            if path and os.path.exists(path):
                                service["build"] = {
                                    "context": str(self.get_docker_context_path()),
                                    "dockerfile": str(path)
                                }
                        if "env_file" not in service:
                            path = self.get_source_path(".env")
                            if path and os.path.exists(path):
                                service["env_file"] = [
                                    str(path)
                                ]
                        container_paths = mount_paths.get(service.get("container_name"), {})
                        if container_paths:
                            volumes = service.setdefault("volumes", [])
                            for container_path in container_paths.values():
                                if container_path not in volumes:
                                    volumes.append(container_path)
                if "networks" in data and isinstance(data["networks"], dict):
                    networks = data["networks"]
                    for name in list(networks.keys()):
                        network = networks[name]
                        network = networks[name]
                        if network is None:
                            network = networks[name] = {}
                        if not isinstance(network, dict):
                            continue
                        network.setdefault("name", self.get_service_name(name))
                return data
        return None

    @cached_property
    def docker_file(self) -> Optional[str]:
        path = self.get_source_path("Dockerfile")
        if os.path.exists(path):
            return self.render_template(path)
        return None

    @cached_property
    def services(self) -> Dict[str, Dict[str, Any]]:
        services: dict = utils.get_item(self.docker_compose, "services")
        if not services or not isinstance(services, dict):
            return {}
        return services

    @cached_property
    def start_hooks(self) -> List[Callable[[], Any]]:
        return []

    @cached_property
    def stop_hooks(self) -> List[Callable[[], Any]]:
        return []

    def on_init(self):
        pass

    def on_starting(self):
        pass

    def on_started(self):
        pass

    def on_stopping(self):
        pass

    def on_stopped(self):
        pass

    def on_removed(self):
        pass

    @subcommand("shell", help="exec into container using command sh")
    @subcommand_argument("-c", "--command", help="shell command")
    @subcommand_argument("--privileged", help="give extended privileges to the command")
    @subcommand_argument("-u", "--user", help="Username or UID (format: \"<name|uid>[:<group|gid>]\")")
    @subcommand_argument("--service", dest="service_name", help="service name")
    def on_exec_shell(self, command: str = None, privileged: bool = False, user: str = None, service_name: str = None):
        service = self.choose_service(service_name)

        options = []
        if privileged:
            options.append("--privileged")
        if user:
            options.append("--user")
            options.append(user)

        if not command:
            commands = []
            for shell in ["/bin/zsh", "/bin/fish", "/bin/bash", "/bin/ash", "/bin/sh"]:
                shell_command = [
                    "if" if len(commands) == 0 else "elif", "[", "-x", shell, "]", ";",
                    "then", shell, ";",
                ]
                commands.extend(shell_command)
            commands.extend(["else", "sh", ";"])
            commands.append("fi")
            commands = ("sh", "-c", utils.list2cmdline(commands))
        else:
            commands = utils.cmdline2list(command)

        return self.manager.create_docker_process(
            "exec", "-it", *options, service.get("container_name"), *commands
        ).call()

    @subcommand("logs", help="fetch the logs of container")
    @subcommand_argument("-f", "--follow",
                         help="follow log output")
    @subcommand_argument("-t", "--timestamps",
                         help="show timestamps")
    @subcommand_argument("-n", "--tail", metavar="string",
                         help="number of lines to show from the end of the logs (default \"all\")")
    @subcommand_argument("--since", metavar="string",
                         help="show logs since timestamp (e.g. \"2013-01-02T13:23:37Z\") or relative (e.g. \"42m\" for 42 minutes)")
    @subcommand_argument("--until", metavar="string",
                         help="show logs before a timestamp (e.g. \"2013-01-02T13:23:37Z\") or relative (e.g. \"42m\" for 42 minutes)")
    @subcommand_argument("--service", dest="service_name", help="service name")
    def on_exec_logs(self, follow: bool = True, tail: str = None, timestamps: bool = True,
                     since: str = None, until: str = None,
                     service_name: str = None):
        service = self.choose_service(service_name)

        options = []
        if follow:
            options.append("--follow")
        if timestamps:
            options.append("--timestamps")
        if tail:
            options.append("--tail")
            options.append(tail)
        if since:
            options.append("--since")
            options.append(since)
        if until:
            options.append("--until")
            options.append(until)
        return self.manager.create_docker_process(
            "logs", *options, service.get("container_name")
        ).call()

    @subcommand("mount", help="mount path")
    @subcommand_argument("source", nargs='?', help="host path")
    @subcommand_argument("target", nargs='?', help="container path")
    @subcommand_argument("-p", "--permission", choices=("ro", "rw"))
    @subcommand_argument("--service", dest="service_name", help="service name")
    def on_mount(self, source: str = None, target: str = None, permission: str = "rw", service_name: str = None):
        if not source or not target:
            if not source and not target:
                with self.settings.open() as settings:
                    result = {}
                    mount_paths = settings.get("mount_paths") or {}
                    for service in self.services.values():
                        container_paths = mount_paths.get(service.get("container_name"), {})
                        if container_paths:
                            result[service.get("container_name")] = list(container_paths.values())
                    if not result:
                        self.logger.info("Not found any mount path")
                        return
                    self.logger.info(yaml.dump(result))
                return
            if not source:
                self.logger.error("Argument error: source is empty")
            if not target:
                self.logger.error("Argument error: target is empty")
            return

        source_path = Path(os.path.expanduser(source)).absolute()
        target_path = PurePosixPath(target).as_posix()
        if not os.path.exists(source_path):
            self.logger.error(f"{source_path} not exists.")
            return

        service = self.choose_service(service_name)
        with self.settings.open() as settings:
            mount_paths = settings.get("mount_paths") or {}
            containers_paths = mount_paths.setdefault(service.get("container_name"), {})
            container_path = f"{source_path}:{target_path}:{permission}"
            if target_path in containers_paths:
                if not confirm(f"{target_path} is mounted: {containers_paths.get(target_path)}, overwrite it?"):
                    self.logger.info(f"cancel")
                    return
            containers_paths[target_path] = container_path
            settings.set("mount_paths", mount_paths)
            self.logger.info(f"add {container_path}")

    @subcommand("umount", help="unmount path")
    @subcommand_argument("--service", dest="service_name", help="service name")
    def on_unmount_file(self, service_name: str = None):
        service = self.choose_service(service_name)
        with self.settings.open() as settings:
            mount_paths = settings.get("mount_paths") or {}
            containers_paths = mount_paths.setdefault(service.get("container_name"), {})
            if not containers_paths:
                self.logger.error("Not found any mount path")
                return
            dest_path = choose(
                "Choose mount path",
                choices=containers_paths
            )
            mount_path = containers_paths.pop(dest_path)
            settings.set("mount_paths", mount_paths)
            self.logger.info(f"remove {mount_path}")

    def get_config(self, key: str, type: "ConfigType" = None, default: Any = __missing__) -> "T":
        return self.manager.config.get(key, type=type, default=default)

    def get_source_path(self, *paths: str) -> Path:
        return utils.join_path(self.root_path, *paths)

    def get_app_path(self, *paths: str, create_parent: bool = False) -> Path:
        path = utils.join_path(self.manager.app_path, self.name, *paths)
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def get_app_data_path(self, *paths: str, create_parent: bool = False) -> Path:
        path = utils.join_path(self.manager.app_data_path, self.name, *paths)
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def get_user_data_path(self, *paths: str, create_parent: bool = False) -> Path:
        path = utils.join_path(self.manager.user_data_path, *paths)
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def get_download_path(self, *paths: str, create_parent: bool = False) -> Path:
        path = utils.join_path(self.manager.download_path, *paths)
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def get_temp_path(self, *paths: str, create_parent: bool = False) -> Path:
        path = utils.join_path(self.manager.temp_path, "container", self.name, *paths)
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def choose_service(self, name: str = None) -> Optional[Dict[str, Any]]:
        services = self.services
        if not services:
            raise ContainerError(f"Not found any service in {self}")
        if name:
            for key, service in services.items():
                if key == name or service.get("container_name") == name:
                    return service
            raise ContainerError(f"Not found service '{name}' in {self}")
        keys = tuple(services.keys())
        key = keys[0] \
            if len(keys) == 1 \
            else choose("Please choose service",
                        choices={key: service.get("container_name") for key, service in self.services.items()},
                        default=keys[0])
        return self.services[key]

    def get_docker_compose_file(self) -> Optional[Path]:
        destination = None
        if self.docker_compose:
            destination = utils.join_path(self.manager.data_path, "compose", f"{self.name}.yml")
            destination.parent.mkdir(parents=True, exist_ok=True)
            utils.write_file(destination, yaml.dump(self.docker_compose))
        return destination

    def get_docker_file_path(self) -> Optional[Path]:
        destination = None
        if self.docker_file:
            destination = utils.join_path(self.manager.data_path, "dockerfile", f"{self.name}.Dockerfile")
            destination.parent.mkdir(parents=True, exist_ok=True)
            utils.write_file(destination, self.docker_file)
        return destination

    def get_docker_context_path(self) -> Path:
        return self.get_source_path()

    def get_service_name(self, key: str) -> str:
        return f"{self.manager.project_name}-{key}"

    def is_depend_on(self, name: str):
        next_items = set(self.dependencies)
        exclude_items = set()
        while next_items:
            if name in next_items:
                return True
            exclude_items.update(next_items)
            current_items = next_items
            next_items = set()
            for next_name in current_items:
                for next_dependency in self.manager.containers[next_name].dependencies:
                    if next_dependency not in exclude_items:
                        next_items.add(next_dependency)
        return False

    def render_template(self, source: PathType, destination: PathType = None, **kwargs: Any):
        config = self.manager.config

        def mkdir(path: PathType) -> str:
            path = config.cast(path, type="path")
            self.start_hooks.append(lambda: os.makedirs(path, mode=0o755, exist_ok=True))
            return path

        def chown(path: PathType, user: str = None) -> str:
            path = config.cast(path, type="path")
            if user:
                uid, gid = utils.get_uid(user), utils.get_gid(user)
                self.start_hooks.append(lambda: self.manager.change_file_owner(path, uid, gid))
            return path

        def chmod(path: PathType, mode: int = 0o755) -> str:
            path = config.cast(path, type="path")
            self.start_hooks.append(lambda: os.chmod(path, mode))
            return path

        context = {
            key: utils.lazy_load(config.get, key)
            for key in config.keys()
        }

        context.update(
            DEBUG=self.manager.debug,

            SOURCE_PATH=utils.lazy_load(self.get_source_path),
            APP_PATH=utils.lazy_load(self.get_app_path),
            APP_DATA_PATH=utils.lazy_load(self.get_app_data_path),
            USER_DATA_PATH=utils.lazy_load(self.get_user_data_path),
            DOWNLOAD_PATH=utils.lazy_load(self.get_download_path),

            manager=self.manager,
            container=self,
            config=config,
            user=self.manager.user,
            docker_user=utils.lazy_load(self.get_config, "DOCKER_USER"),

            mkdir=mkdir,
            chown=chown,
            chmod=chmod,
        )

        context.update(kwargs)

        environment = Environment(
            loader=FileSystemLoader(Path(source).parent)
        )
        environment.filters.update(
            mkdir=mkdir,
            chown=chown,
            chmod=chmod,
        )

        try:
            self.logger.debug(f"{self} render template {source} to {destination or 'memory'}")
            template = environment.from_string(utils.read_file(source, text=True))
            result = template.render(context)
            if destination:
                utils.write_file(destination, result)
            return result

        except TemplateError as e:
            raise ContainerTemplateError(e)

    def __repr__(self):
        return f"Container<{self.name}>"


class SimpleContainer(BaseContainer):

    def __init__(self, manager: "ContainerManager", root_path: str):
        super().__init__(
            manager,
            root_path,
            name=os.path.basename(root_path)
        )
