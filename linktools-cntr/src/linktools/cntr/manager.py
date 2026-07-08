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
import contextlib
import inspect
import json
import os
import pathlib
import shutil
from typing import TYPE_CHECKING

from dulwich.errors import NotGitRepository

from linktools import utils
from linktools.system import get_gid, get_lan_ip, get_machine, get_system, get_uid, get_user
from linktools.core import (
    ConfigField, ChainProvider, PromptProvider, LazyProvider, AliasProvider,
)
from linktools.decorator import cached_property
from linktools.errors import GitDivergedError
from linktools.git import GitRepository, GitSyncPolicy
from linktools.types import MISSING
from linktools.runtime import import_module_file, popen

from . import _migrate
from .container import BaseContainer, SimpleContainer, ContainerError
from ..capabilities.cntr import __cap_cntr__

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from typing import Any
    from linktools.core import Environ
    from linktools.types import PathType
    from linktools.runtime import Process
    from .context import EventContext


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

        self.name = name or self.environ.name
        self.environ = environ
        self.logger = environ.get_logger("container")

        self.env_config = self.environ.wrap_config(namespace="container", env_prefix="")
        self.env_config.update_defaults(**self.configs)

        self.docker_container_name = "container.py"
        self.docker_compose_names = ("compose.yaml", "compose.yml", "docker-compose.yaml", "docker-compose.yml")

        self._setting_cache = {}

    @property
    def configs(self) -> "dict[str, Any]":
        return dict(
            HOST=ConfigField(name="HOST", provider=ChainProvider(
                PromptProvider("HOST"),
                LazyProvider(lambda r: get_lan_ip()),
            )),
            DOCKER_HOST="/var/run/docker.sock",

            COMPOSE_PROJECT_NAME=self.name,
            SERVICE_RESTART_POLICY="unless-stopped",
            SERVICE_LOG_DRIVER="json-file",
            SERVICE_LOG_MAX_SIZE="10m",

            DOCKER_USER=ConfigField(name="DOCKER_USER", provider=ChainProvider(
                PromptProvider("DOCKER_USER"),
            ), default=os.environ.get("SUDO_USER", self.user).replace(" ", "")),
            DOCKER_UID=ConfigField(name="DOCKER_UID", provider=LazyProvider(
                lambda r: get_uid(r.get("DOCKER_USER", type=str)),
            )),
            DOCKER_GID=ConfigField(name="DOCKER_GID", provider=LazyProvider(
                lambda r: get_gid(r.get("DOCKER_USER", type=str)),
            )),
            DOCKER_TYPE=ConfigField(name="DOCKER_TYPE", default="docker", provider=(
                ChainProvider(
                    AliasProvider("CONTAINER_TYPE"),
                    PromptProvider("DOCKER_TYPE", choices=["docker", "docker-rootless"]),
                ) if self.system == "linux" and os.getuid() != 0 else None
            )),

            DOCKER_APP_PATH=ConfigField(name="DOCKER_APP_PATH", cast="path", provider=ChainProvider(
                PromptProvider("DOCKER_APP_PATH"),
            ), default=str(self.data_path.joinpath("app"))),
            DOCKER_APP_DATA_PATH=ConfigField(name="DOCKER_APP_DATA_PATH", cast="path",
                                             provider=AliasProvider("DOCKER_APP_PATH")),
            DOCKER_USER_DATA_PATH=ConfigField(name="DOCKER_USER_DATA_PATH", cast="path", provider=ChainProvider(
                PromptProvider("DOCKER_USER_DATA_PATH"),
            ), default=str(self.data_path.joinpath("user_data"))),
            DOCKER_DOWNLOAD_PATH=ConfigField(name="DOCKER_DOWNLOAD_PATH", cast="path", provider=ChainProvider(
                PromptProvider("DOCKER_DOWNLOAD_PATH"),
            ), default=str(self.data_path.joinpath("download"))),
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
        _migrate.migrate_legacy_container_settings(
            self._persistent_store, self.data_path, self.setting_path, self.logger
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
        for container in self._load_containers():
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

    def _load_containers(self) -> "list[BaseContainer]":
        containers: "list[BaseContainer]" = []

        self.logger.debug(f"Load containers from assets")
        asset_path = __cap_cntr__.get_asset_path("containers")
        for container in self._walk_containers(asset_path, max_level=1):
            containers.append(container)

        for url, meta in self.get_all_repos().items():
            self.logger.debug(f"Load containers from repository `{url}`")
            repo_path = meta.get("repo_path")
            if not repo_path or not os.path.exists(repo_path) or not os.path.isdir(repo_path):
                self.logger.warning(f"Repository `{url}` not found, skip.")
                continue
            for container in self._walk_containers(repo_path, max_level=2):
                containers.append(container)

        return containers

    def _walk_containers(self, path: "PathType", max_level: int):
        if not os.path.isdir(path):
            return
        yield from self._load_container(path)
        if max_level <= 0:
            return
        for name in os.listdir(path):
            yield from self._walk_containers(
                os.path.join(path, name),
                max_level - 1
            )

    def _load_container(self, path: "PathType"):
        container_path = os.path.join(path, self.docker_container_name)
        if os.path.exists(container_path):
            try:
                name = path.replace(os.sep, ".")
                module = import_module_file(name, container_path)
                for key, value in module.__dict__.items():
                    if isinstance(value, type) and issubclass(value, BaseContainer):
                        if not value.__abstract__:
                            container = value(self, path)
                            self.logger.debug(f"Load container {container.name} in {path}")
                            self._callback(container.on_init)
                            yield container
                            return
            except Exception as e:
                self.logger.warning(f"Failed to load container from `{path}`: {e}")
                return

        for compose_name in self.docker_compose_names:
            compose_path = os.path.join(path, compose_name)
            if os.path.exists(compose_path):
                container = SimpleContainer(self, path)
                self.logger.debug(f"Load container {container.name} in {path}")
                self._callback(container.on_init)
                yield container
                return

    def get_installed_containers(self, resolve: bool = True) -> "list[BaseContainer]":
        with self.environ.locks.process_lock("cntr:settings"):
            containers = self._load_installed_containers()
        if resolve:
            containers = self.resolve_depend_containers(containers)
        return containers

    def resolve_depend_containers(self, containers: "Iterable[BaseContainer]") -> "list[BaseContainer]":
        result: "list[BaseContainer]" = []
        visited: "set[BaseContainer]" = set()
        visiting: "set[BaseContainer]" = set()

        def visit(container: "BaseContainer"):
            if container in visited:
                return
            if container in visiting:
                raise ContainerError(f"Circular dependency detected for container {container}")
            visiting.add(container)
            for dependency in container.dependencies:
                if dependency not in self.containers:
                    raise ContainerError(f"Dependency container {dependency} not found")
                visit(self.containers[dependency])
            visiting.remove(container)
            visited.add(container)
            result.append(container)

        for container in sorted(containers, key=lambda o: (o.order, o.name)):
            visit(container)
        return result

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
        with self.environ.locks.process_lock("cntr:settings"):
            result = set()
            for name in names:
                container = self.containers.get(name, None)
                if container:
                    result.add(container)
            containers = self._load_installed_containers(reload=True)
            containers.extend(result)
            self._dump_installed_containers(containers)
            return list(result)

    def remove_installed_containers(self, *names: str, force: bool = False) -> "list[BaseContainer]":
        with self.environ.locks.process_lock("cntr:settings"):
            containers = self._load_installed_containers(reload=True)

            result = set()
            remove_names = set(names)
            for name in set(names):
                if name not in self.containers:
                    continue
                for container in containers:
                    if not container.is_depend_on(name):
                        continue
                    if container.name in remove_names:
                        continue
                    if force:
                        remove_names.add(container.name)
                    elif container not in remove_names:
                        raise ContainerError(
                            f"{container} depends on {self.containers[name]}, "
                            f"cannot remove {self.containers[name]}"
                        )

            for name in remove_names:
                container = self.containers.get(name, None)
                if container and container in containers:
                    result.add(container)
                    containers.remove(container)

            self._dump_installed_containers(containers)

            return list(result)

    def _load_installed_containers(self, reload: bool = False) -> "list[BaseContainer]":
        result = set()
        for name in self._load_setting("INSTALLED_CONTAINERS", reload=reload, default=[]):
            if name in self.containers:
                result.add(self.containers[name])
        return list(result)

    def _dump_installed_containers(self, containers: "Iterable[BaseContainer]") -> None:
        self._dump_setting("INSTALLED_CONTAINERS", list(set([container.name for container in containers])))

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

    @contextlib.contextmanager
    def notify_start(self, context: "EventContext"):
        for container in context.target_containers:
            self._callback(container.on_check, context)

        for container in context.target_containers:
            self._callback(container.on_starting, context)

        for container in context.target_containers:
            if container.start_hooks:
                for hook in container.start_hooks:
                    self._callback(hook)

        if self.start_hooks:
            for hook in self.start_hooks:
                self._callback(hook)

        yield

        for container in reversed(context.target_containers):
            self._callback(container.on_started, context)

    @contextlib.contextmanager
    def notify_stop(self, context: "EventContext"):
        for container in reversed(context.target_containers):
            self._callback(container.on_stopping, context)

        yield

        for container in context.target_containers:
            self._callback(container.on_stopped, context)
            if container.stop_hooks:
                for hook in container.stop_hooks:
                    self._callback(hook)

        if self.stop_hooks:
            for hook in self.stop_hooks:
                self._callback(hook)

    @contextlib.contextmanager
    def notify_remove(self, context: "EventContext"):
        yield

        if context.is_full_containers:
            with self.environ.locks.process_lock("cntr:settings"):
                running_containers = self._load_running_containers()
                all_containers = {*context.containers, *running_containers}
                for container in running_containers:
                    if container not in context.containers:
                        # A removed container is no longer in the installed list, so its
                        # `configs` defaults were never registered in env_config. Register
                        # them here so on_removed can read its own configs without failing.
                        self.env_config.update_defaults(**container.configs)
                        self._callback(container.on_removed, context)
                        all_containers.remove(container)
                self._dump_running_containers(all_containers)

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
        if privilege:
            if self.system in ("darwin", "linux") and self.uid != 0:
                proxy_keys = ("http_proxy", "https_proxy", "all_proxy", "no_proxy")
                preserve_keys = [*[e.lower() for e in proxy_keys], *[e.upper() for e in proxy_keys]]
                preserve_env = [key for key in preserve_keys if key in os.environ]
                sudo_args = ["sudo"]
                if preserve_env:
                    sudo_args.append(f"--preserve-env={','.join(preserve_env)}")
                sudo_args.extend(args)
                return popen(*sudo_args, **kwargs)
        return popen(*args, **kwargs)

    def create_docker_process(
            self,
            *args,
            privilege: bool = None,
            **kwargs
    ) -> "Process":
        commands = []
        if self.container_type in ("docker", "docker-rootless"):
            commands.extend(["docker"])
            if privilege is None:
                privilege = self.container_type == "docker"
        elif self.container_type == "podman":
            commands.extend(["podman"])
        else:
            raise ContainerError(f"Invalid container type: {self.container_type}")
        return self.create_process(*commands, *args, privilege=privilege, **kwargs)

    def create_docker_compose_process(
            self,
            containers: "Iterable[BaseContainer]",
            *args: str,
            privilege: bool = None,
            **kwargs: "Any"
    ) -> "Process":
        options = []
        for container in containers:
            path = container.get_docker_compose_file()
            if path and os.path.exists(path):
                options.extend(["--file", path])
        options.extend(["--project-name", self.project_name])
        return self.create_docker_process("compose", *options, *args, privilege=privilege, **kwargs)

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
        if not shutil.which("chown"):
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
        return self._load_setting("INSTALLED_REPOS", default={})

    def add_repo(self, url: str, branch: str = None, force: bool = False):
        with self.environ.locks.process_lock("cntr:repo"):
            repos = self._load_setting("INSTALLED_REPOS", reload=True, default={})

            def ensure_repo_not_exist(key):
                if key not in repos:
                    return
                if not force:
                    raise ContainerError(f"Repository `{key}` already exists.")
                self._remove_repo_file(repos.pop(key))
                self._dump_setting("INSTALLED_REPOS", repos)

            if url.startswith("http://") or url.startswith("https://") or \
                    url.startswith("ssh://") or url.startswith("git@"):
                ensure_repo_not_exist(url)
                self.logger.info(f"Add git repository: {url}")
                repo_name = utils.guess_file_name(url)
                repo_path = self._choose_repo_path(repo_name)
                GitRepository.clone(self.environ, url, repo_path, branch)
                repos[url] = dict(type="git", repo_path=repo_path, repo_name=repo_name)

            else:
                path = os.path.abspath(os.path.expanduser(url))
                if not os.path.exists(path) or not os.path.isdir(path):
                    raise ContainerError(f"Invalid local path: {url}")

                ensure_repo_not_exist(path)
                self.logger.info(f"Add local repository: {path}")
                repo_name = utils.guess_file_name(path)
                repo_path = self._choose_repo_path(repo_name)
                os.symlink(path, repo_path, target_is_directory=True)
                repos[path] = dict(type="local", repo_path=repo_path, repo_name=repo_name)

            self._dump_setting("INSTALLED_REPOS", repos)

    def update_repos(self, branch: str = None, reset: bool = False):
        for url, meta in self.get_all_repos().items():
            repo_type = meta.get("type", None)
            repo_path = meta.get("repo_path", None)
            if not repo_path:
                continue

            if repo_type == "git" and not os.path.exists(repo_path):
                self.logger.info(f"Update git repository: {url}")
                GitRepository.clone(self.environ, url, repo_path, branch)
                continue

            if not os.path.exists(repo_path):
                continue

            try:
                repo = GitRepository(self.environ, repo_path)
            except NotGitRepository:
                self.logger.debug(f"Invalid git repository, skip: {url}")
                continue

            if repo_type == "git":
                self.logger.info(f"Update git repository: {url}")

            is_stash = False
            try:
                if repo.is_dirty():
                    if not reset:
                        self.logger.info(f"Repository `{repo_path}` is dirty, stash changes before pull")
                        is_stash = True
                        repo.git.stash()
                    else:
                        self.logger.warning(f"Repository `{repo_path}` is dirty, reset to HEAD")
                        repo.git.reset(hard=True)

                if branch:
                    if branch in repo.heads:
                        self.logger.info(f"Checkout branch `{branch}` in repository `{repo_path}`")
                        repo.git.checkout(branch)
                    else:
                        self.logger.info(f"Branch `{branch}` not found in repository `{repo_path}`, create and checkout")
                        new_branch = repo.create_head(branch)
                        new_branch.checkout()

                try:
                    repo.sync(policy=GitSyncPolicy.RESET_TO_REMOTE if reset
                              else GitSyncPolicy.FAST_FORWARD_ONLY)
                except GitDivergedError:
                    if reset:
                        raise
                    self.logger.warning(
                        f"Repository `{url}` has diverged from the remote, force resetting ..."
                    )
                    repo.sync(policy=GitSyncPolicy.RESET_TO_REMOTE)

            finally:
                if is_stash:
                    self.logger.info(f"Repository `{repo_path}` is updated, pop stashed changes")
                    repo.git.stash("pop")

    def remove_repo(self, url: str):
        with self.environ.locks.process_lock("cntr:repo"):
            repos = self._load_setting("INSTALLED_REPOS", reload=True, default={})
            if url not in repos:
                raise ContainerError(f"Repository `{url}` not found.")
            self._remove_repo_file(repos.pop(url))
            self._dump_setting("INSTALLED_REPOS", repos)

    def _choose_repo_path(self, name: str):
        index = 0
        path = os.path.join(self._repo_path, name)
        while os.path.lexists(path):
            path = os.path.join(self._repo_path, f"{name}_{index}")
            index += 1
        return path

    def _remove_repo_file(self, repo: "dict[str, str]"):
        repo_path = repo.get("repo_path", None)
        if repo_path and os.path.lexists(repo_path):
            if os.path.islink(repo_path):
                self.logger.info(f"Remove link {repo_path}")
                os.unlink(repo_path)
            elif os.path.isdir(repo_path):
                self.logger.info(f"Remove directory {repo_path}")
                shutil.rmtree(repo_path, ignore_errors=True)

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
