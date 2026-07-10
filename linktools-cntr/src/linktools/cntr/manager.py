#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : repo.py 
@time    : 2024/3/22
@site    : https://github.com/ice-black-tea
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
import inspect
import os
import pathlib
from typing import TYPE_CHECKING

from linktools import utils
from linktools.system import get_gid, get_lan_ip, get_machine, get_system, get_uid, get_user
from linktools.core import (
    ConfigField, PromptProvider, LazyProvider, AliasProvider,
)
from linktools.decorator import cached_property
from linktools.types import MISSING

from . import _migrate
from .container import BaseContainer, ContainerError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from typing import Any
    from linktools.core import Environ
    from linktools.runtime import Process
    from linktools.types import PathType
    from .context import EventContext
    from .registry.resolver import ContainerResolver
    from .registry.loader import ContainerLoader
    from .runtime.compose import ComposeRunner
    from .runtime.process import RuntimeProcessFactory
    from .lifecycle.dispatcher import LifecycleDispatcher
    from .state.running import RunningStateStore
    from .state.installed import InstalledStateStore
    from .repo.store import RepoStore


class ContainerManager:

    def __init__(self, environ: "Environ", name: str = "aio"):  # all_in_one
        self.user = get_user()
        self.uid = get_uid()
        self.gid = get_gid()
        self.system = get_system()
        self.machine = get_machine()

        self.environ = environ
        self.name = name or self.environ.name
        self.logger = environ.get_logger("container")

        self.env_config = self.environ.wrap_config(namespace="container", env_prefix="")
        self.env_config.update_defaults(**self.configs)

        self.docker_container_name = "container.py"
        self.docker_compose_names = ("compose.yaml", "compose.yml", "docker-compose.yaml", "docker-compose.yml")

        # Trigger legacy -> ConfigStore migration now, before any config or
        # setting is read/written. State/repo stores also touch self._migrated
        # on every access, but "config set/list/edit" reach env_config
        # directly without ever going through a state/repo store, so relying
        # on that alone would let those commands run against pre-migration data.
        try:
            self._migrated
        except Exception as exc:
            self.logger.warning(f"Failed to run legacy settings migration: {exc}")

    @property
    def configs(self) -> "dict[str, Any]":
        return dict(
            HOST=ConfigField.chain(
                PromptProvider(),
                LazyProvider(lambda r: get_lan_ip()),
            ),
            DOCKER_HOST="/var/run/docker.sock",

            COMPOSE_PROJECT_NAME=self.name,
            SERVICE_RESTART_POLICY="unless-stopped",
            SERVICE_LOG_DRIVER="json-file",
            SERVICE_LOG_MAX_SIZE="10m",

            DOCKER_USER=ConfigField.chain(
                PromptProvider(cached=True),
                default=os.environ.get("SUDO_USER", self.user).replace(" ", ""),
            ),
            DOCKER_UID=ConfigField(provider=LazyProvider(
                lambda r: get_uid(r.get("DOCKER_USER", type=str)),
            )),
            DOCKER_GID=ConfigField(provider=LazyProvider(
                lambda r: get_gid(r.get("DOCKER_USER", type=str)),
            )),
            DOCKER_TYPE=ConfigField.chain(
                AliasProvider("CONTAINER_TYPE"),
                PromptProvider(choices=["docker", "docker-rootless"], cached=True),
                default="docker",
            ) if self.system == "linux" and os.getuid() != 0 else ConfigField(default="docker"),

            DOCKER_APP_PATH=ConfigField.chain(
                PromptProvider(cached=True), cast="path", default=str(self.data_path.joinpath("app")),
            ),
            DOCKER_APP_DATA_PATH=ConfigField(cast="path", provider=AliasProvider("DOCKER_APP_PATH")),
            DOCKER_USER_DATA_PATH=ConfigField.chain(
                PromptProvider(cached=True), cast="path", default=str(self.data_path.joinpath("user_data")),
            ),
            DOCKER_DOWNLOAD_PATH=ConfigField.chain(
                PromptProvider(cached=True), cast="path", default=str(self.data_path.joinpath("download")),
            ),
        )

    @property
    def debug(self) -> bool:
        return os.environ.get("DEBUG", self.environ.debug)

    @cached_property
    def container_type(self) -> str:
        return self.env_config.get("DOCKER_TYPE", type=str)

    @cached_property
    def container_host(self) -> str:
        host = self.env_config.get("DOCKER_HOST", type=str)
        if host:
            left, sep, right = host.partition("://")
            return right or left
        return "/var/run/docker.sock"

    @cached_property
    def host(self) -> str:
        return self.env_config.get("HOST", type=str)

    @cached_property
    def project_name(self) -> str:
        return self.env_config.get("COMPOSE_PROJECT_NAME")

    @cached_property
    def root_path(self):
        return pathlib.Path(os.path.dirname(__file__))

    @cached_property
    def app_path(self):
        return self.env_config.get("DOCKER_APP_PATH")

    @cached_property
    def app_data_path(self):
        return self.env_config.get("DOCKER_APP_DATA_PATH")

    @cached_property
    def data_path(self):
        return self.environ.get_data_path("container")

    @cached_property
    def temp_path(self):
        return self.environ.get_temp_path("container")

    @cached_property
    def setting_path(self):
        path = utils.join_path(self.data_path, "setting")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @cached_property
    def _persistent_store(self):
        """Persistent user state: INSTALLED_CONTAINERS / INSTALLED_REPOS."""
        return self.environ.config_store

    @cached_property
    def _transient_ns(self):
        """Transient settings (RUNNING_CONTAINERS, ...) in the cache store."""
        return self.environ.cache.namespace("cntr")

    @cached_property
    def _migrated(self):
        # See _migrate.py for what this consolidates and why. A failed
        # migration is not cached, so later access (e.g. from a state/repo
        # store) retries it.
        _migrate.migrate_legacy_container_settings(
            self._persistent_store, self.data_path, self.setting_path, self.logger,
        )
        return True

    @cached_property
    def containers(self) -> "dict[str, BaseContainer]":
        result = dict()
        for container in self.loader.load_all():
            if container.name in result:
                self.logger.debug(f"Container `{container.name}` already exists, overwrite.")
            result[container.name] = container
        # Raw persisted names only -- resolving them to container objects here
        # would recurse back into this same property via InstalledStateStore.
        for name in self.installed_state.load_names():
            if name not in result:
                self.logger.warning(f"Not found installed container `{name}`, skip.")
        return result

    @cached_property
    def start_hooks(self) -> "list[Callable[[], Any]]":
        return []

    @cached_property
    def stop_hooks(self) -> "list[Callable[[], Any]]":
        return []

    @cached_property
    def compose_runner(self) -> "ComposeRunner":
        # CLI and per-container exec route through this so both share one
        # docker-compose argument builder.
        from .runtime.compose import ComposeRunner
        return ComposeRunner(self)

    @cached_property
    def resolver(self) -> "ContainerResolver":
        from .registry.resolver import ContainerResolver
        return ContainerResolver(self)

    @cached_property
    def loader(self) -> "ContainerLoader":
        from .registry.loader import ContainerLoader
        return ContainerLoader(self)

    @cached_property
    def runtime(self) -> "RuntimeProcessFactory":
        from .runtime.process import RuntimeProcessFactory
        return RuntimeProcessFactory(self)

    @cached_property
    def lifecycle(self) -> "LifecycleDispatcher":
        from .lifecycle.dispatcher import LifecycleDispatcher
        return LifecycleDispatcher(self)

    @cached_property
    def running_state(self) -> "RunningStateStore":
        from .state.running import RunningStateStore
        return RunningStateStore(self)

    @cached_property
    def installed_state(self) -> "InstalledStateStore":
        from .state.installed import InstalledStateStore
        return InstalledStateStore(self)

    @cached_property
    def repo_store(self) -> "RepoStore":
        from .repo.store import RepoStore
        return RepoStore(self)

    def get_installed_containers(self, resolve: bool = True) -> "list[BaseContainer]":
        return self.installed_state.get(resolve=resolve)

    def resolve_depend_containers(self, containers: "Iterable[BaseContainer]") -> "list[BaseContainer]":
        return self.resolver.resolve_dependencies(containers)

    def prepare_installed_containers(self) -> "list[BaseContainer]":
        self.logger.debug(f"Load container type: {self.container_type}")  # 加载容器类型
        containers = self.get_installed_containers(resolve=True)
        if not containers:
            raise ContainerError("No container installed")
        for container in self.containers.values():
            container.enable = container in containers
        for container in reversed(containers):
            self.env_config.update_defaults(**container.configs)
        for container in containers:
            self._callback(func=container.on_prepare)
        for container in containers:
            if container.docker_file and self.debug:  # 加载每个容器的dockerfile
                self.logger.debug(f"Generate Dockerfile for {container.name}")
            if container.docker_compose and self.debug:  # 加载每个容器的docker-compose.yml
                self.logger.debug(f"Generate docker-compose.yml for {container.name}")
            if container.exposes and self.debug:
                self.logger.debug(f"Load exposes for {container.name}")
        return containers

    def add_installed_containers(self, *names: str) -> "list[BaseContainer]":
        return self.installed_state.add(*names)

    def remove_installed_containers(self, *names: str, force: bool = False) -> "list[BaseContainer]":
        return self.installed_state.remove(*names, force=force)

    def get_running_containers(self):
        with self.environ.locks.process_lock("cntr:settings"):
            return self._load_running_containers()

    def _load_running_containers(self):
        # A failed migration is not cached, so retry it on every access.
        self._migrated
        result = set()
        for name in self._transient_ns.get("RUNNING_CONTAINERS", []) or []:
            if name in self.containers:
                result.add(self.containers[name])
        return list(result)

    def _dump_running_containers(self, containers: "Iterable[BaseContainer]") -> None:
        self._migrated
        self._transient_ns.set(
            "RUNNING_CONTAINERS", list({container.name for container in containers}))

    def notify_start(self, context: "EventContext"):
        return self.lifecycle.notify_start(context)

    def notify_stop(self, context: "EventContext"):
        return self.lifecycle.notify_stop(context)

    def notify_remove(self, context: "EventContext"):
        return self.lifecycle.notify_remove(context)

    def _callback(self, func, context: "EventContext" = MISSING):
        if self.environ.debug:
            self.logger.debug(f"Callback {func}")
        if context is MISSING:
            return func()
        sig = inspect.signature(func)
        if len(sig.parameters) == 0:
            return func()
        else:
            return func(context)

    def create_process(
            self,
            *args,
            privilege: bool = None,
            **kwargs
    ) -> "Process":
        return self.runtime.create_process(*args, privilege=privilege, **kwargs)

    def create_docker_process(
            self,
            *args,
            privilege: bool = None,
            **kwargs
    ) -> "Process":
        return self.runtime.create_docker_process(*args, privilege=privilege, **kwargs)

    def create_docker_compose_process(
            self,
            containers: "Iterable[BaseContainer]",
            *args: str,
            privilege: bool = None,
            **kwargs: "Any"
    ) -> "Process":
        return self.runtime.create_docker_compose_process(containers, *args, privilege=privilege, **kwargs)

    def change_file_owner(self, path: "PathType", user: str, recursive: bool = False) -> None:
        return self.runtime.chown(path, user, recursive=recursive)

    def change_file_mode(self, path: "PathType", mode: int = 0o755, recursive: bool = False) -> None:
        return self.runtime.chmod(path, mode, recursive=recursive)

    def get_all_repos(self) -> "dict[str, dict[str, str]]":
        return self.repo_store.get_all()

    def add_repo(self, url: str, branch: str = None, force: bool = False):
        return self.repo_store.add(url, branch=branch, force=force)

    def update_repos(self, branch: str = None, reset: bool = False):
        return self.repo_store.update(branch=branch, reset=reset)

    def remove_repo(self, url: str):
        return self.repo_store.remove(url)
