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
import functools
import json
import os
import os.path
import pathlib
import shutil
from typing import TYPE_CHECKING, Dict, Any, List, Union, Callable, Tuple, Set

from linktools import utils
from linktools.core import Config
from linktools.decorator import cached_property
from linktools.types import PathType, FileCache
from .container import BaseContainer, SimpleContainer, ContainerError
from .repository import Repository
from ..capabilities.cntr import __capability__

if TYPE_CHECKING:
    from linktools.core import Environ


class ContainerManager:

    def __init__(self, environ: "Environ", name: str = "aio"):  # all_in_one
        self.user = utils.get_user()
        self.uid = utils.get_uid()
        self.gid = utils.get_gid()
        self.system = utils.get_system()
        self.machine = utils.get_machine()

        self.environ = environ
        self.logger = environ.get_logger("container")

        self.config = self.environ.wrap_config(namespace="container", env_prefix="")
        self.config.update_defaults(
            COMPOSE_PROJECT_NAME=name or self.environ.name,
            DOCKER_USER=Config.Prompt(default=os.environ.get("SUDO_USER", self.user).replace(" ", ""), cached=True),
            DOCKER_UID=Config.Lazy(lambda cfg: utils.get_uid(cfg.get("DOCKER_USER", type=str))),
            DOCKER_GID=Config.Lazy(lambda cfg: utils.get_gid(cfg.get("DOCKER_USER", type=str))),
            SERVICE_RESTART_POLICY="unless-stopped",
            SERVICE_LOG_DRIVER="json-file",
            SERVICE_LOG_MAX_SIZE="10m",
        )

        self.docker_container_name = "container.py"
        self.docker_compose_names = ("compose.yaml", "compose.yml", "docker-compose.yaml", "docker-compose.yml")

        self._setting_cache = {}

    @property
    def debug(self) -> bool:
        return os.environ.get("DEBUG", self.environ.debug)

    @cached_property
    def container_type(self) -> str:
        choices = ["docker", "docker-rootless", "podman"] \
            if self.system in ("darwin", "linux") and os.getuid() != 0 \
            else ["docker", "podman"]
        return self.config.get(
            "CONTAINER_TYPE",
            type=str,
            default=Config.Prompt(choices=choices, cached=True),
        )

    @cached_property
    def container_host(self) -> str:
        default = "/var/run/docker.sock"
        host = self.config.get(
            "DOCKER_HOST",
            type=str,
            default=default
        )
        if host:
            left, sep, right = host.partition("://")
            return right or left
        return default

    @cached_property
    def host(self) -> str:
        return self.config.get(
            "HOST",
            type=str,
            default=Config.Prompt(default=utils.get_lan_ip())
        )

    @cached_property
    def project_name(self) -> str:
        return self.config.get("COMPOSE_PROJECT_NAME")

    @cached_property
    def root_path(self):
        return pathlib.Path(os.path.dirname(__file__))

    @cached_property
    def app_path(self):
        return self.config.get(
            "DOCKER_APP_PATH",
            type="path",
            default=Config.Prompt(
                default=Config.Lazy(
                    lambda cfg: self.environ.get_data_path("container", "app", create_parent=True)
                ),
                cached=True,
            )
        )

    @cached_property
    def app_data_path(self):
        return self.config.get(
            "DOCKER_APP_DATA_PATH",
            type="path",
            default=Config.Prompt(
                default=Config.Lazy(
                    lambda cfg: self.environ.get_data_path("container", "app_data", create_parent=True)
                ),
                cached=True,
            )
        )

    @cached_property
    def user_data_path(self):
        return self.config.get(
            "DOCKER_USER_DATA_PATH",
            type="path",
            default=Config.Prompt(
                default=Config.Lazy(
                    lambda cfg: self.environ.get_data_path("container", "user_data", create_parent=True)
                ),
                cached=True,
            )
        )

    @cached_property
    def download_path(self):
        return self.config.get(
            "DOCKER_DOWNLOAD_PATH",
            type="path",
            default=Config.Prompt(
                default=Config.Lazy(
                    lambda cfg: self.environ.get_data_path("container", "download", create_parent=True)
                ),
                cached=True,
            )
        )

    @cached_property
    def data_path(self):
        return self.environ.get_data_path("container", create_parent=True)

    @cached_property
    def temp_path(self):
        return self.environ.get_temp_path("container", create_parent=True)

    @cached_property
    def setting_path(self):
        path = utils.join_path(self.data_path, "setting")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @cached_property
    def _settings(self):
        settings = FileCache(self.setting_path / "manager")

        config_path = self.environ.get_data_path("container", "config", "containers.yml")
        repo_path = self.environ.get_data_path("container", "repo", "repo.json")
        repo_lock = self.environ.get_data_path("container", "repo", "repo.lock")

        if os.path.isfile(config_path) or os.path.isfile(repo_path):
            with settings.open() as data:
                if os.path.isfile(config_path):
                    self.logger.warning("Found old config file, try to migrate.")
                    try:
                        with open(config_path) as fd:
                            data.set("INSTALLED_CONTAINERS", json.load(fd))
                        utils.remove_file(config_path.parent)
                    except Exception as e:
                        self.logger.warning(f"Failed to load old config file: {e}")

                if os.path.isfile(repo_path):
                    self.logger.warning("Found old repo file, try to migrate.")
                    try:
                        with open(repo_path) as fd:
                            data.set("INSTALLED_REPOS", json.load(fd))
                        utils.remove_file(repo_path)
                        utils.remove_file(repo_lock)
                    except:
                        self.logger.warning(f"Failed to load old repo file: {e}")

        return settings

    @cached_property
    def _repo_path(self):
        return self.environ.get_data_path("container", "repo", create_parent=True)

    @cached_property
    def containers(self) -> Dict[str, BaseContainer]:
        result = dict()
        for container in self._load_containers():
            if container.name in result:
                self.logger.debug(f"Container `{container.name}` already exists, overwrite.")
            result[container.name] = container
        for name in self._load_setting("INSTALLED_CONTAINERS", default=[]):
            if name not in result:
                self.logger.warning(f"Not found installed container `{name}`, skip.")
        return result

    def _load_containers(self) -> List[BaseContainer]:
        containers: List[BaseContainer] = []

        self.logger.debug(f"Load containers from assets")
        asset_path = __capability__.get_asset_path("containers")
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

    def _walk_containers(self, path: PathType, max_level: int):
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

    def _load_container(self, path: PathType):
        container_path = os.path.join(path, self.docker_container_name)
        if os.path.exists(container_path):
            try:
                name = path.replace(os.sep, ".")
                module = utils.import_module_file(name, container_path)
                for key, value in module.__dict__.items():
                    if isinstance(value, type) and issubclass(value, BaseContainer):
                        if not value.__abstract__:
                            container = value(self, path)
                            self.logger.debug(f"Load container {container.name} in {path}")
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
                yield container
                return

    def get_installed_containers(self, resolve: bool = True) -> List[BaseContainer]:
        with self._settings.lock():
            containers = self._load_installed_containers()
        if resolve:
            containers = self.resolve_depend_containers(containers)
        return containers

    def resolve_depend_containers(self, containers: List[BaseContainer]) -> List[BaseContainer]:
        order = lambda o: o() if callable(o) else o
        result: Dict[BaseContainer, Union[int, Callable[[], int]]] = dict()
        container_queue: Set[BaseContainer] = set(containers)
        while container_queue:
            container = container_queue.pop()
            result.setdefault(container, container.order)
            for dependency in container.dependencies:
                if dependency not in self.containers:
                    raise ContainerError(f"Dependency container {dependency} not found")
                depend_container = self.containers[dependency]
                if depend_container not in result:
                    result.setdefault(depend_container, depend_container.order)
                    container_queue.add(depend_container)
                if order(result[depend_container]) >= order(result[container]):
                    result[depend_container] = functools.partial(lambda o: order(result[o]) - 1, container)
        return sorted(result, key=lambda o: (order(result[o]), o.order, o.name))

    def prepare_installed_containers(self) -> List[BaseContainer]:
        self.logger.debug(f"Load container type: {self.container_type}")  # 加载容器类型
        containers = self.get_installed_containers(resolve=True)
        if not containers:
            raise ContainerError("No container installed")
        for container in self.containers.values():
            container.enable = container in containers
        for container in reversed(containers):
            self.config.update_defaults(**container.configs)
        for container in containers:
            container.on_init()
        for container in containers:
            if container.docker_file and self.debug:  # 加载每个容器的dockerfile
                self.logger.debug(f"Generate Dockerfile for {container.name}")
            if container.docker_compose and self.debug:  # 加载每个容器的docker-compose.yml
                self.logger.debug(f"Generate docker-compose.yml for {container.name}")
            if container.exposes and self.debug:
                self.logger.debug(f"Load exposes for {container.name}")
        return containers

    def add_installed_containers(self, *names: str) -> List[BaseContainer]:
        with self._settings.lock():
            result = set()
            for name in names:
                container = self.containers.get(name, None)
                if container:
                    result.add(container)
            containers = self._load_installed_containers(reload=True)
            containers.extend(result)
            self._dump_installed_containers(containers)
            return list(result)

    def remove_installed_containers(self, *names: str, force: bool = False) -> List[BaseContainer]:
        with self._settings.lock():
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

    def _load_installed_containers(self, reload: bool = False) -> List[BaseContainer]:
        result = set()
        for name in self._load_setting("INSTALLED_CONTAINERS", reload=reload, default=[]):
            if name in self.containers:
                result.add(self.containers[name])
        return list(result)

    def _dump_installed_containers(self, containers: List[BaseContainer]) -> None:
        self._dump_setting("INSTALLED_CONTAINERS", list(set([container.name for container in containers])))

    def create_process(
            self,
            *args,
            privilege: bool = None,
            **kwargs
    ) -> utils.Process:
        if privilege:
            if self.system in ("darwin", "linux") and self.uid != 0:
                envs = ("http_proxy", "https_proxy", "all_proxy", "no_proxy")
                args = [
                    "sudo",
                    f"--preserve-env={','.join([e.lower() for e in envs] + [e.upper() for e in envs])}",
                    *args
                ]
        return utils.create_process(*args, **kwargs)

    def create_docker_process(
            self,
            *args,
            privilege: bool = None,
            **kwargs
    ) -> utils.Process:
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
            containers: List[BaseContainer],
            *args: str,
            privilege: bool = None,
            **kwargs: Any
    ) -> utils.Process:
        options = []
        for container in containers:
            path = container.get_docker_compose_file()
            if path and os.path.exists(path):
                options.extend(["--file", path])
        options.extend(["--project-name", self.project_name])
        return self.create_docker_process("compose", *options, *args, privilege=privilege, **kwargs)

    def change_file_owner(self, path: PathType, uid: int, gid: int, force: bool = False,
                          privilege: bool = False) -> None:
        if not os.path.exists(path):
            self.logger.debug(f"Path not found: {path}")
            return
        if not shutil.which("chown"):
            self.logger.debug("Command `chown` not found")
            return
        stat = os.stat(path)
        if force or stat.st_uid != uid or stat.st_gid != gid:
            try:
                self.create_process(
                    "chown", "-R", f"{uid}:{gid}", path,
                    privilege=privilege or self.uid != stat.st_uid or self.uid != uid
                ).check_call()
            except Exception as e:
                self.logger.warning(f"Failed to change owner of {path}")
                raise e

    def get_all_repos(self) -> Dict[str, Dict[str, str]]:
        return self._load_setting("INSTALLED_REPOS", default={})

    def add_repo(self, url: str, branch: str = None, force: bool = False):
        with self._settings.lock("repo"):
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
                Repository.clone_with_progress(url, repo_path, branch)
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

    def update_repos(self, force: bool = False):
        for url, meta in self.get_all_repos().items():
            repo_type = meta.get("type", None)
            repo_path = meta.get("repo_path", None)
            if repo_type == "git" and repo_path:
                self.logger.info(f"Update git repository: {url}")
                if not os.path.exists(repo_path):
                    Repository.clone_with_progress(url, repo_path)
                    continue
                repo = Repository(repo_path)
                if repo.is_dirty():
                    if not force:
                        raise ContainerError(f"Repository `{repo_path}` is dirty")
                    self.logger.warning(f"Repository `{repo_path}` is dirty, reset to HEAD")
                    repo.git.reset(hard=True)
                repo.update_with_progress()

    def remove_repo(self, url: str):
        with self._settings.lock("repo"):
            repos = self._load_setting("INSTALLED_REPOS", reload=True, default={})
            if url not in repos:
                raise ContainerError(f"Repository `{url}` not found.")
            self._remove_repo_file(repos.pop(url))
            self._dump_setting("INSTALLED_REPOS", repos)

    def _choose_repo_path(self, name: str):
        index = 0
        path = os.path.join(self._repo_path, name)
        while os.path.exists(path):
            path = os.path.join(self._repo_path, f"{name}_{index}")
            index += 1
        return path

    def _remove_repo_file(self, repo: Dict[str, str]):
        repo_path = repo.get("repo_path", None)
        if repo_path and os.path.exists(repo_path):
            if os.path.islink(repo_path):
                self.logger.info(f"Remove link {repo_path}")
                os.unlink(repo_path)
            elif os.path.isdir(repo_path):
                self.logger.info(f"Remove directory {repo_path}")
                shutil.rmtree(repo_path, ignore_errors=True)

    def _load_setting(self, key: str, reload: bool = False, default: Any = None) -> Union[Dict, List, Tuple]:
        if reload:
            self._setting_cache.pop(key, None)
        elif key in self._setting_cache:
            return self._setting_cache[key]
        with self._settings.open() as data:
            result = self._setting_cache[key] = data.get(key, default)
            return result

    def _dump_setting(self, key: str, setting: Union[Dict, List, Tuple]):
        self._setting_cache.pop(key, None)
        with self._settings.open() as data:
            data.set(key, setting)
