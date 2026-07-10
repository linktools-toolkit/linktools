#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import textwrap
from typing import TYPE_CHECKING

from linktools import utils
from linktools.cli import subcommand, subcommand_argument
from linktools.cli.argparse import BooleanOptionalAction
from linktools.core import ConfigField
from linktools.decorator import cached_property
from linktools.errors import Error
from linktools.rich import choose
from linktools.runtime import lazy_load
from linktools.types import MISSING
from linktools.utils import get_md5
from ._container import compose as _compose
from ._container import exec as _exec
from ._container import template as _template
from ._container.expose import ExposeCategory, ExposeLink, ExposeMixin
from ._container.nginx import NginxMixin

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path
    from typing import Any
    from linktools.types import T, ConfigType, ConfigKeyType, PathType
    from .manager import ContainerManager
    from .context import EventContext


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

    def __init__(self, manager: "ContainerManager", root_path: "PathType", name: str = None):
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
    def configs(self) -> "dict[str, Any]":
        return {}

    @property
    def extend_configs(self) -> "dict[str, Any]":
        return {}

    @property
    def exposes(self) -> "Iterable[ExposeLink]":
        return []

    @property
    def settings(self):
        """Return this container's operational cache namespace."""
        return self.manager.environ.cache.namespace("cntr:app:" + self.name)

    @cached_property
    def docker_compose(self) -> "dict[str, Any] | None":
        return _compose.load_docker_compose(self)

    @cached_property
    def docker_file(self) -> "str | None":
        return _compose.load_docker_file(self)

    @cached_property
    def services(self) -> "dict[str, dict[str, Any]]":
        return _compose.get_services(self)

    @cached_property
    def start_hooks(self) -> "list[Callable[[], Any]]":
        return []

    @cached_property
    def stop_hooks(self) -> "list[Callable[[], Any]]":
        return []

    @cached_property
    def _rendered_hook_keys(self) -> "set[tuple]":
        # Idempotency ledger for hook registration: tracks which (action,
        # target, ...) keys have already produced a hook so re-rendering a
        # template -- or rendering the same path in compose and Dockerfile --
        # does not register duplicate start/stop hooks.
        return set()

    def add_start_hook(self, key: "tuple", hook: "Callable[[], Any]") -> None:
        """Register a start hook once per key."""
        if key in self._rendered_hook_keys:
            return
        self._rendered_hook_keys.add(key)
        self.start_hooks.append(hook)

    def on_init(self):
        pass

    def on_prepare(self):
        pass

    def on_check(self, context: "EventContext"):
        pass

    def on_starting(self, context: "EventContext"):
        pass

    def on_started(self, context: "EventContext"):
        pass

    def on_stopping(self, context: "EventContext"):
        pass

    def on_stopped(self, context: "EventContext"):
        pass

    def on_removed(self, context: "EventContext"):
        pass

    @subcommand("up", help="deploy this container")
    @subcommand_argument("--build", action=BooleanOptionalAction, help="build images before starting")
    @subcommand_argument("--pull", action=BooleanOptionalAction,
                         help="always attempt to pull a newer version of the image")
    def on_exec_up(self, build: bool = True, pull: bool = False):
        return _exec.up(self, build=build, pull=pull)

    @subcommand("restart", help="restart this container")
    @subcommand_argument("--build", action=BooleanOptionalAction, help="build images before starting")
    @subcommand_argument("--pull", action=BooleanOptionalAction,
                         help="always attempt to pull a newer version of the image")
    def on_exec_restart(self, build: bool = True, pull: bool = False):
        return _exec.restart(self, build=build, pull=pull)

    @subcommand("down", help="stop this container")
    def on_exec_down(self):
        return _exec.down(self)

    @subcommand("config", help="show docker compose config for this container")
    def on_exec_config(self):
        return _exec.config(self)

    @subcommand("shell", help="exec into container using command sh")
    @subcommand_argument("-c", "--command", help="shell command")
    @subcommand_argument("--privileged", help="give extended privileges to the command")
    @subcommand_argument("-u", "--user", help="Username or UID (format: \"<name|uid>[:<group|gid>]\")")
    @subcommand_argument("--service", dest="service_name", help="service name")
    def on_exec_shell(self, command: str = None, privileged: bool = False, user: str = None, service_name: str = None):
        return _exec.shell(self, command=command, privileged=privileged, user=user, service_name=service_name)

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
        return _exec.logs(self, follow=follow, tail=tail, timestamps=timestamps,
                          since=since, until=until, service_name=service_name)

    @subcommand("mount", help="mount path")
    @subcommand_argument("source", nargs='?', help="host path")
    @subcommand_argument("target", nargs='?', help="container path")
    @subcommand_argument("-p", "--permission", choices=("ro", "rw"))
    @subcommand_argument("--service", dest="service_name", help="service name")
    def on_mount(self, source: str = None, target: str = None, permission: str = "rw", service_name: str = None):
        return _exec.mount(self, source=source, target=target, permission=permission, service_name=service_name)

    @subcommand("umount", help="unmount path")
    @subcommand_argument("--service", dest="service_name", help="service name")
    def on_unmount_file(self, service_name: str = None):
        return _exec.umount(self, service_name=service_name)

    def _resolve_config_key(self, key: "ConfigKeyType") -> str:
        """Accept either a plain field name or a ``ConfigField`` to define.

        Lets a container reference a one-off field (e.g. a nginx domain with
        its own fallback) directly at the call site -- ``get_config(ConfigField(
        name="X", default=...))`` -- instead of also declaring a ``configs``
        property purely to give the field a home. Defining is idempotent
        (``ConfigSchema.define`` just overwrites the same name), so repeated
        calls are safe.
        """
        if isinstance(key, ConfigField):
            if not key.name:
                raise ValueError("ConfigField passed as a config key must have a name")
            self.manager.env_config.define(key)
            return key.name
        return key

    def get_config(self, key: "ConfigKeyType", type: "ConfigType" = None, default: "Any" = MISSING) -> "T":
        return self.manager.env_config.get(self._resolve_config_key(key), type=type, default=default)

    def get_config_later(self, key: "ConfigKeyType", type: "ConfigType" = None, default: "Any" = MISSING) -> "T":
        return lazy_load(self.manager.env_config.get, self._resolve_config_key(key), type=type, default=default)

    def _make_exec_context(self, commands) -> "EventContext":
        from .context import EventContext

        containers = self.manager.get_installed_containers(resolve=True)
        if self not in containers:
            raise ContainerError(f"{self} is not installed")

        context = EventContext()
        context.commands = [commands] if isinstance(commands, str) else list(filter(None, commands))
        context.containers = containers
        context.target_containers = [self]
        context.is_full_containers = False
        return context

    def get_source_path(self, *paths: str) -> "Path":
        return utils.join_path(self.root_path, *paths)

    def get_app_path(self, *paths: str, create_parent: bool = False) -> "Path":
        path = utils.join_path(self.manager.app_path, self.name, *paths)
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def get_app_data_path(self, *paths: str, create_parent: bool = False) -> "Path":
        path = utils.join_path(self.manager.app_data_path, self.name, *paths)
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def get_temp_path(self, *paths: str, create_parent: bool = False) -> "Path":
        path = utils.join_path(self.manager.temp_path, "container", self.name, *paths)
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def choose_service(self, name: str = None) -> "dict[str, Any] | None":
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

    def get_docker_compose_file(self) -> "Path | None":
        return _compose.write_docker_compose_file(self)

    def get_docker_file_path(self) -> "Path | None":
        return _compose.write_docker_file(self)

    def get_docker_context_path(self) -> "Path":
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

    def render_template(self, source: "PathType", destination: "PathType" = None, **kwargs: "Any"):
        return _template.render_template(self, source, destination=destination, **kwargs)

    def __repr__(self):
        return f"Container<{self.name}>"


class SourceContainer(BaseContainer):
    __abstract__ = True

    @property
    def _source_url(self):
        raise NotImplementedError()

    @property
    def _source_path(self):
        raise NotImplementedError()

    def _handle_source_file(self, source: "PathType", destination: "PathType"):
        # Archive extraction rejects traversal, absolute paths and unsafe links.
        from linktools.utils import safe_extract
        safe_extract(source, destination)

    @cached_property
    def _context_path(self):
        name = get_md5(self._source_url)
        source_path = self.get_app_path("source", f"{name}.in")
        dest_path = self.get_app_path("source", f"{name}.out")

        def init_source_code():
            if not os.path.isdir(dest_path):
                file = self.manager.environ.get_url_file(self._source_url)
                file.save(source_path.parent, source_path.name)
                os.makedirs(dest_path, exist_ok=True)
                try:
                    self._handle_source_file(source_path, dest_path)
                except:
                    utils.remove_file(source_path)
                    utils.remove_file(dest_path)
                    raise

        self.start_hooks.append(init_source_code)
        return os.path.join(dest_path, self._source_path)

    def get_docker_context_path(self):
        return self._context_path

    def on_starting(self, context: "EventContext"):
        if "pull" in context.commands:
            utils.remove_file(self.get_app_path("source"))

    def on_removed(self, context: "EventContext"):
        utils.remove_file(self.get_app_path("source"))


class SimpleContainer(BaseContainer):

    def __init__(self, manager: "ContainerManager", root_path: str):
        super().__init__(
            manager,
            root_path,
            name=os.path.basename(root_path)
        )
