#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``ContainerManager``: the facade owning config, container discovery,
lifecycle, state, repos, and every other cntr subsystem."""
import json
import os
import pathlib
from typing import TYPE_CHECKING

from linktools import utils
from linktools.system import get_gid, get_lan_ip, get_machine, get_system, get_uid, get_user
from linktools.core import AliasProvider, ConfigField, LazyProvider, PromptProvider
from linktools.decorator import cached_property

from .container import BaseContainer, ContainerError, NoContainerInstalledError

if TYPE_CHECKING:
    from typing import Any
    from linktools.core import CacheNamespace, ConfigStore, Environ
    from .registry.registry import ContainerResolver
    from .registry.loader import ContainerLoader
    from .operations import ComposeOperations
    from .runtime.compose import ComposeRunner
    from .runtime.process import RuntimeProcessFactory
    from .runtime.structured import StructuredCommandRunner
    from .runtime.inspect import DockerInspector
    from .lifecycle.dispatcher import LifecycleDispatcher
    from .lifecycle.hooks import HookListView, HookRegistry
    from .state.running import RunningStateStore
    from .state.installed import InstalledStateStore
    from .repo.service import RepoService
    from .artifacts import ArtifactIndex
    from .execution.planner import ExecutionPlanner


def describe_origin(container: "BaseContainer") -> str:
    """Safe, credential-free description of where a container came from --
    ``builtin`` or a repository's ``repo_name`` (never its URL, which may
    embed a Git credential), plus its filesystem root_path. Used to
    identify the two sides of a duplicate-container-name error."""
    context = container.repo_context
    if context is None:
        return "container:%s" % id(container.env_config)
    origin = "builtin" if context.builtin else (context.repo_name or "repo")
    root_path = context.root_path
    return f"{origin} ({root_path})" if root_path else origin


class ContainerManager:

    def __init__(self, environ: "Environ", name: str = "aio"):  # all_in_one
        self.environ = environ
        self.name = name or environ.name
        self.logger = environ.get_logger("container")

        self.env_config = self.environ.build_config("container", "")
        self.env_config.update_defaults(**self.configs)

        # Populated by the `containers` cached_property -- structured load
        # failures (import/on_init errors) from the last time containers
        # were discovered, for callers (e.g. Doctor) that want to report
        # them instead of only the log-only warning.
        self.container_load_errors: "list[Any]" = []

        self.docker_container_name = "container.py"
        self.docker_compose_names = ("compose.yaml", "compose.yml", "docker-compose.yaml", "docker-compose.yml")

    @property
    def user(self) -> str:
        return get_user()

    @property
    def uid(self) -> int:
        return get_uid()

    @property
    def gid(self) -> int:
        return get_gid()

    @property
    def system(self) -> str:
        return get_system()

    @property
    def machine(self) -> str:
        return get_machine()

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
        # Routed through env_config (cast=bool) instead of a raw
        # os.environ.get(...) -- DEBUG=0/DEBUG=false must parse as False,
        # not the truthy-any-non-empty-string behavior a bare os.environ
        # read (or Python's own bool(str)) would give.
        return self.env_config.get("DEBUG", type=bool, default=self.environ.debug)

    # Plain @property, not @cached_property -- these few config-derived
    # values are cheap to resolve (env_config.get() is itself memoized per
    # revision), and caching them here would let a set_config/persist/reload
    # within the same process keep returning a stale value forever, since
    # nothing ever invalidates a manager-level cached_property.
    @property
    def container_type(self) -> str:
        return self.env_config.get("DOCKER_TYPE", type=str)

    @property
    def container_host(self) -> str:
        host = self.env_config.get("DOCKER_HOST", type=str)
        if host:
            left, sep, right = host.partition("://")
            return right or left
        return "/var/run/docker.sock"

    @property
    def host(self) -> str:
        return self.env_config.get("HOST", type=str)

    @property
    def project_name(self) -> str:
        return self.env_config.get("COMPOSE_PROJECT_NAME")

    @cached_property
    def root_path(self):
        return pathlib.Path(os.path.dirname(__file__))

    @property
    def app_path(self):
        return self.env_config.get("DOCKER_APP_PATH")

    @property
    def app_data_path(self):
        return self.env_config.get("DOCKER_APP_DATA_PATH")

    @cached_property
    def data_path(self):
        return self.environ.get_data_path("container")

    @cached_property
    def temp_path(self):
        return self.environ.get_temp_path("container")

    @cached_property
    def settings(self) -> "ConfigStore":
        """Persistent store for all of cntr's own state: INSTALLED_CONTAINERS/
        INSTALLED_REPOS (top-level keys) plus each container's own operational
        settings (namespaced ``cntr:app:<name>`` keys, see BaseContainer.settings).
        """
        from ._migrate import migrate_legacy_settings

        return migrate_legacy_settings(
            self, 
            self.environ.build_config_store("container.json")
        )

    @cached_property
    def cache(self) -> "CacheNamespace":
        """Transient settings (RUNNING_CONTAINERS, ...) in the cache store."""
        return self.environ.cache.namespace("cntr")

    @cached_property
    def containers(self) -> "dict[str, BaseContainer]":
        result = dict()
        load_result = self.loader.load_all()
        self.container_load_errors = load_result.errors
        for container in load_result.containers:
            existing = result.get(container.name)
            if existing is not None:
                raise ContainerError(
                    "Duplicate container name %r: %s and %s"
                    % (container.name, describe_origin(existing), describe_origin(container)))
            result[container.name] = container
        # Raw persisted names only -- resolving them to container objects here
        # would recurse back into this same property via InstalledStateStore.
        errors_by_name = {e.expected_name: e for e in load_result.errors if e.expected_name}
        for name in self.installed_state.load_names():
            if name not in result:
                error = errors_by_name.get(name)
                if error is not None:
                    self.logger.warning(
                        f"Installed container `{name}` failed to load from `{error.path}`: {error.message}")
                else:
                    self.logger.warning(f"Not found installed container `{name}`, skip.")
        return result

    @cached_property
    def hooks(self) -> "HookRegistry":
        from .lifecycle.hooks import HookRegistry
        return HookRegistry(owner=self, scope="manager")

    @cached_property
    def start_hooks(self) -> "HookListView":
        from .lifecycle.hooks import HookPhase
        return self.hooks.legacy_view(HookPhase.BEFORE_START)

    @cached_property
    def stop_hooks(self) -> "HookListView":
        from .lifecycle.hooks import HookPhase
        return self.hooks.legacy_view(HookPhase.AFTER_STOP)

    @cached_property
    def compose_runner(self) -> "ComposeRunner":
        # CLI and per-container exec route through this so both share one
        # docker-compose argument builder.
        from .runtime.compose import ComposeRunner
        return ComposeRunner(self)

    @cached_property
    def compose_operations(self) -> "ComposeOperations":
        # Root up/restart/down and the `compose` command both dispatch
        # through this so they can never drift from each other.
        from .operations import ComposeOperations
        return ComposeOperations(self)

    @cached_property
    def resolver(self) -> "ContainerResolver":
        from .registry.registry import ContainerResolver
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
    def structured_runner(self) -> "StructuredCommandRunner":
        from .runtime.structured import StructuredCommandRunner
        return StructuredCommandRunner(self)

    @cached_property
    def docker_inspector(self) -> "DockerInspector":
        from .runtime.inspect import DockerInspector
        return DockerInspector(self)

    @cached_property
    def artifact_index(self) -> "ArtifactIndex":
        from .artifacts import ArtifactIndex
        return ArtifactIndex(self)

    @cached_property
    def planner(self) -> "ExecutionPlanner":
        from .execution.planner import ExecutionPlanner
        return ExecutionPlanner(self)

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
    def repos(self) -> "RepoService":
        from .repo.service import RepoService
        return RepoService(self)

    def load_installed_config_metadata(self) -> "list[BaseContainer]":
        """Load installed containers and register their own config fields,
        without running any container's ``on_prepare()`` (arbitrary
        third-party file writes, network access, hook registration) or
        touching ``docker_file``/``docker_compose``.

        Safe for anything that only needs config metadata: config
        set/get/list/explain/validate/reload, Root ``list``, Plan, Doctor.
        Returns ``[]`` when nothing is installed instead of raising -- see
        ``prepare_installed_containers`` for the raising, side-effectful
        variant real execution (up/down/restart/exec) needs.
        """
        containers = self.installed_state.get(resolve=True)
        for container in self.containers.values():
            container.enable = container in containers
        for container in reversed(containers):
            container.register_configs()
        return containers

    def prepare_installed_containers(self) -> "list[BaseContainer]":
        self.logger.debug(f"Load container type: {self.container_type}")  # 加载容器类型
        containers = self.load_installed_config_metadata()
        if not containers:
            raise NoContainerInstalledError("No container installed")
        for container in containers:
            container.on_prepare()
        for container in containers:
            if container.docker_file and self.debug:  # 加载每个容器的dockerfile
                self.logger.debug(f"Generate Dockerfile for {container.name}")
            if container.docker_compose and self.debug:  # 加载每个容器的docker-compose.yml
                self.logger.debug(f"Generate docker-compose.yml for {container.name}")
            if container.exposes and self.debug:
                self.logger.debug(f"Load exposes for {container.name}")
        return containers
