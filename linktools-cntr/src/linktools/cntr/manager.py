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
import shutil
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
    from linktools.types import PathType
    from linktools.runtime import Process
    from .context import EventContext
    from .registry.resolver import ContainerResolver
    from .registry.loader import ContainerLoader
    from .runtime.compose import ComposeRunner
    from .runtime.process import RuntimeProcessFactory
    from .lifecycle.dispatcher import LifecycleDispatcher
    from .state.running import RunningStateStore
    from .state.installed import InstalledStateStore
    from .repo.store import RepoStore


def _is_chown_supported(system: str = None) -> bool:
    """Return whether chown/chmod on a host path is reflected inside a container
    bind-mounting that path. Docker Desktop's VM-backed bind mounts on macOS and
    Windows don't honor host-side ownership/permission changes, so this is only
    true on Linux."""
    return (system or get_system()) == "linux"


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

        self._setting_cache = {}

        # Trigger legacy -> ConfigStore migration now, before any config or
        # setting is read/written. _load_setting also touches self._migrated,
        # but "config set/list/edit" reach env_config directly without ever
        # calling _load_setting, so relying on that alone would let those
        # commands run against pre-migration data.
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
        """Persistent user state (spec §8.5): INSTALLED_CONTAINERS / REPOS."""
        return self.environ.config_store

    @cached_property
    def _transient_ns(self):
        """Transient settings (RUNNING_CONTAINERS, ...) in the cache store."""
        return self.environ.cache.namespace("cntr")

    @cached_property
    def _migrated(self):
        # One-time legacy -> ConfigStore migration. Runs on first
        # access of any setting; idempotent.
        #
        # The legacy <data>/.config/<name>.cfg ini file (CONTAINER.CACHE.* +
        # MAIN.CACHE.*) is migrated by Environ.config_store itself, the first
        # time anything touches the persistent store -- which happens before
        # this point (self.env_config in __init__ already forced it via
        # wrap_config -> PersistentSource(self.config_store, ...)). So by the
        # time _migrated runs, that file is already gone; only cntr's other
        # three legacy sources (containers.yml / repo.json / FileCache shelve)
        # remain here.
        _migrate.migrate_legacy_container_settings(
            self._persistent_store, self.data_path, self.setting_path, self.logger,
        )
        return True

    @cached_property
    def _repo_path(self):
        path = self.data_path.joinpath("repo")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @cached_property
    def containers(self) -> "dict[str, BaseContainer]":
        result = dict()
        for container in self.loader.load_all():
            if container.name in result:
                self.logger.debug(f"Container `{container.name}` already exists, overwrite.")
            result[container.name] = container
        for name in self._load_setting("INSTALLED_CONTAINERS", default=[]):
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
        # Unified docker-compose command assembly (refactor spec Phase 2).
        # CLI and per-container exec route through this so the two paths share
        # one argument builder while each preserves its exact original commands.
        from .runtime.compose import ComposeRunner
        return ComposeRunner(self)

    @cached_property
    def resolver(self) -> "ContainerResolver":
        # Dependency resolution behind the facade (refactor spec Phase 4).
        from .registry.resolver import ContainerResolver
        return ContainerResolver(self)

    @cached_property
    def loader(self) -> "ContainerLoader":
        # Container discovery/import behind the facade (refactor spec Phase 4).
        from .registry.loader import ContainerLoader
        return ContainerLoader(self)

    @cached_property
    def runtime(self) -> "RuntimeProcessFactory":
        # docker/podman/compose process creation behind the facade (Phase 4).
        from .runtime.process import RuntimeProcessFactory
        return RuntimeProcessFactory(self)

    @cached_property
    def lifecycle(self) -> "LifecycleDispatcher":
        # Lifecycle hook dispatch behind the facade (refactor spec Phase 4).
        from .lifecycle.dispatcher import LifecycleDispatcher
        return LifecycleDispatcher(self)

    @cached_property
    def running_state(self) -> "RunningStateStore":
        # Persisted running-container state behind the facade (refactor spec Phase 5).
        from .state.running import RunningStateStore
        return RunningStateStore(self)

    @cached_property
    def installed_state(self) -> "InstalledStateStore":
        # Persisted installed-container set behind the facade (refactor spec Phase 4).
        from .state.installed import InstalledStateStore
        return InstalledStateStore(self)

    @cached_property
    def repo_store(self) -> "RepoStore":
        # External repository management behind the facade (refactor spec Phase 4).
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
        result = set()
        for name in self._load_setting("RUNNING_CONTAINERS", reload=True, default=[]):
            if name in self.containers:
                result.add(self.containers[name])
        return list(result)

    def _dump_running_containers(self, containers: "Iterable[BaseContainer]") -> None:
        self._dump_setting("RUNNING_CONTAINERS", list(set([container.name for container in containers])))

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
        path = self.env_config.cast(path, type="path")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")
        if not _is_chown_supported(self.system):
            self.logger.debug(f"Skip chown of {path} on {self.system}")
            return
        if not shutil.which("chown"):
            self.logger.debug("Command `chown` not found")
            return
        args = ["chown"]
        if recursive:
            args.append("-R")
        uid, gid = get_uid(user), get_gid(user)
        args.extend([f"{uid}:{gid}", str(path)])
        try:
            stat = os.stat(path)
            self.create_process(
                *args,
                privilege=self.uid != stat.st_uid or self.uid != uid
            ).check_call()
        except Exception as e:
            self.logger.warning(f"Failed to chown of {path}")
            raise e

    def change_file_mode(self, path: "PathType", mode: int = 0o755, recursive: bool = False) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")
        if not _is_chown_supported(self.system):
            self.logger.debug(f"Skip chmod of {path} on {self.system}")
            return
        if not shutil.which("chmod"):
            self.logger.debug("Command `chmod` not found")
            return
        args = ["chmod"]
        if recursive:
            args.append("-R")
        args.extend([oct(mode)[2:], str(path)])
        try:
            stat = os.stat(path)
            self.create_process(
                *args,
                privilege=self.uid != stat.st_uid
            ).check_call()
        except Exception as e:
            self.logger.warning(f"Failed to chmod of {path}")
            raise e

    def get_all_repos(self) -> "dict[str, dict[str, str]]":
        return self.repo_store.get_all()

    def add_repo(self, url: str, branch: str = None, force: bool = False):
        return self.repo_store.add(url, branch=branch, force=force)

    def update_repos(self, branch: str = None, reset: bool = False):
        return self.repo_store.update(branch=branch, reset=reset)

    def remove_repo(self, url: str):
        return self.repo_store.remove(url)

    def _load_setting(self, key: str, reload: bool = False, default: "Any" = None) -> "dict | list | tuple":
        self._migrated  # ensure legacy data has been moved into ConfigStore
        if reload:
            self._setting_cache.pop(key, None)
        elif key in self._setting_cache:
            return self._setting_cache[key]
        if key in _migrate.PERSISTENT_KEYS:
            result = self._persistent_store.get(key, default)
        else:
            result = self._transient_ns.get(key, default)
        # Existence is decided by row presence in the new stores (no falsy drop),
        # so the old `if result is None: result = default` workaround is gone.
        self._setting_cache[key] = result
        return result

    def _dump_setting(self, key: str, setting: "dict | list | tuple"):
        self._setting_cache.pop(key, None)
        if key in _migrate.PERSISTENT_KEYS:
            self._persistent_store.set(key, setting)
        else:
            self._transient_ns.set(key, setting)
