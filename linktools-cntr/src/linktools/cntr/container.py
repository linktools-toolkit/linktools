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
from linktools.rich import choose
from linktools.runtime import lazy_load
from linktools.types import MISSING
from linktools.utils import get_md5
from .errors import ContainerError, ContainerTemplateError, NoContainerInstalledError
from ._container import actions as _actions
from ._container import compose as _compose
from ._container import template as _template

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path
    from typing import AbstractSet, Any, Mapping
    from linktools.core import Config, ConfigNamespace, Environ
    from linktools.types import T, ConfigType, ConfigKeyType, PathType
    from .ext import Integrations
    from .manager import ContainerManager
    from .context import OperationContext
    from .repo.context import RepositoryConfigContext
    from .lifecycle.hooks import HookRegistry
    from .runtime.compose import ComposeRunner
    from .runtime.process import RuntimeProcessFactory
    from .lifecycle.dispatcher import LifecycleDispatcher
    from .state.running import RunningStateStore


class AbstractMetaClass(type):

    def __new__(mcs, name, bases, namespace):
        if "__abstract__" not in namespace:
            namespace["__abstract__"] = False
        return super().__new__(mcs, name, bases, namespace)


class BaseContainer(metaclass=AbstractMetaClass):
    __abstract__ = True

    def __init__(self, manager: "ContainerManager", root_path: "PathType", name: str = None):
        self.manager = manager
        self.logger = manager.logger
        self.root_path = root_path
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
        self._repo_context = None

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
    def enable(self, value: bool) -> None:
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
    def integrations(self) -> "Integrations":
        return []

    @cached_property
    def settings(self) -> "ConfigNamespace":
        """Return this container's persistent operational settings namespace.

        Backed by ``manager.settings`` (``cntr.json``, not
        ``environ.cache``): unlike cache, this is never age-swept by
        ``clean_temp_files``, since real user configuration lives here (e.g.
        ``ct-cntr mount``'s persisted paths) -- data an unused-for-N-days
        sweep must never silently drop.
        """
        namespace_key = "cntr:app:" + self.name
        new_ns = self.manager.settings.namespace(namespace_key)
        from linktools.core import Config
        if Config.is_read_only_resolution():
            return new_ns

        # One-time migration: this namespace used to live in the cache store
        # (regenerable/age-swept by design); anything a pre-existing
        # installation already persisted there must move to the permanent
        # store instead of silently disappearing under the next sweep.
        old_ns = self.environ.cache.namespace(namespace_key)
        old_keys = old_ns.keys()
        if old_keys and not new_ns.keys():
            with new_ns.transaction() as ns:
                for key in old_keys:
                    ns.set(key, old_ns.get(key))
            for key in old_keys:
                old_ns.delete(key)

        return new_ns

    # -- Direct access to shared manager services -------------------------
    # Lightweight `@property`, not `@cached_property`: the instance itself is
    # already cached on the manager, so these never construct a second copy.

    @property
    def environ(self) -> "Environ":
        return self.manager.environ

    @property
    def repo_context(self) -> "RepositoryConfigContext | None":
        """Where this container came from -- ``None`` if never attached by
        ContainerLoader (e.g. a container built directly in a test)."""
        return self._repo_context

    @repo_context.setter
    def repo_context(self, context: "RepositoryConfigContext | None") -> None:
        self._repo_context = context

    @property
    def env_config(self) -> "Config":
        # Every container, builtin or third-party, resolves fields through
        # the manager's own shared Config.
        return self.manager.env_config

    @property
    def runtime(self) -> "RuntimeProcessFactory":
        return self.manager.runtime

    @property
    def compose_runner(self) -> "ComposeRunner":
        return self.manager.compose_runner

    @property
    def lifecycle(self) -> "LifecycleDispatcher":
        return self.manager.lifecycle

    @property
    def running_state(self) -> "RunningStateStore":
        return self.manager.running_state

    @property
    def project_name(self) -> str:
        return self.manager.project_name

    @property
    def host(self) -> str:
        return self.manager.host

    @property
    def user(self) -> str:
        return self.manager.user

    @property
    def containers(self) -> "dict[str, BaseContainer]":
        return self.manager.containers

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
    def hooks(self) -> "HookRegistry":
        from .lifecycle.hooks import HookRegistry
        return HookRegistry(owner=self, scope="container")

    def add_start_hook(self, key: "tuple", hook: "Callable[[], Any]", **kwargs: "Any") -> None:
        """Register a BEFORE_START hook once per key (idempotent re-render)."""
        from .lifecycle.hooks import HookPhase
        self.hooks.register(HookPhase.BEFORE_START, hook, key=key, **kwargs)

    def add_stop_hook(self, key: "tuple", hook: "Callable[[], Any]", **kwargs: "Any") -> None:
        """Register an AFTER_STOP hook once per key (idempotent re-render)."""
        from .lifecycle.hooks import HookPhase
        self.hooks.register(HookPhase.AFTER_STOP, hook, key=key, **kwargs)

    def on_init(self) -> None:
        pass


    def get_build_revision(self, service: str) -> "str | None":
        """Optional local build identity; never refresh remote inputs."""
        return None

    def get_runtime_requirements(self, required: "AbstractSet[str]") -> "Mapping[str, Iterable[str]]":
        """Declare native providers needed by the selected project services."""
        return {}

    def on_check(self, context: "OperationContext") -> None:
        pass

    def on_starting(self, context: "OperationContext") -> None:
        pass

    def on_started(self, context: "OperationContext") -> None:
        pass

    def on_stopping(self, context: "OperationContext") -> None:
        pass

    def on_stopped(self, context: "OperationContext") -> None:
        pass

    def on_removed(self, context: "OperationContext") -> None:
        pass

    @subcommand("up", help="deploy this container")
    @subcommand_argument("--pull", action=BooleanOptionalAction,
                         help="always attempt to pull a newer version of the image")
    def on_exec_up(self, pull: bool = False) -> None:
        return _actions.up(self, pull=pull)

    @subcommand("restart", help="restart this container")
    @subcommand_argument("--pull", action=BooleanOptionalAction,
                         help="always attempt to pull a newer version of the image")
    def on_exec_restart(self, pull: bool = False) -> None:
        return _actions.restart(self, pull=pull)

    @subcommand("down", help="stop this container")
    def on_exec_down(self) -> None:
        return _actions.down(self)

    @subcommand("config", help="show docker compose config for this container")
    def on_exec_config(self) -> "dict[str, Any] | None":
        return _actions.config(self)

    @subcommand("shell", help="exec into container using command sh")
    @subcommand_argument("-c", "--command", help="shell command")
    @subcommand_argument("--privileged", help="give extended privileges to the command")
    @subcommand_argument("-u", "--user", help="Username or UID (format: \"<name|uid>[:<group|gid>]\")")
    @subcommand_argument("--service", dest="service_name", help="service name")
    def on_exec_shell(self, command: "str | None" = None, privileged: bool = False, user: "str | None" = None, service_name: "str | None" = None) -> int:
        return _actions.shell(self, command=command, privileged=privileged, user=user, service_name=service_name)

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
    def on_exec_logs(self, follow: bool = True, tail: "str | None" = None, timestamps: bool = True,
                     since: "str | None" = None, until: "str | None" = None,
                     service_name: "str | None" = None) -> int:
        return _actions.logs(self, follow=follow, tail=tail, timestamps=timestamps,
                          since=since, until=until, service_name=service_name)

    @subcommand("mount", help="mount path")
    @subcommand_argument("source", nargs='?', help="host path")
    @subcommand_argument("target", nargs='?', help="container path")
    @subcommand_argument("-p", "--permission", choices=("ro", "rw"))
    @subcommand_argument("--service", dest="service_name", help="service name")
    def on_mount(self, source: "str | None" = None, target: "str | None" = None, permission: str = "rw", service_name: "str | None" = None) -> None:
        return _actions.mount(self, source=source, target=target, permission=permission, service_name=service_name)

    @subcommand("umount", help="unmount path")
    @subcommand_argument("--service", dest="service_name", help="service name")
    def on_unmount_file(self, service_name: "str | None" = None) -> None:
        return _actions.umount(self, service_name=service_name)

    def register_configs(self) -> None:
        """Register this container's own ``configs`` onto its Config."""
        for key, spec in self.configs.items():
            self.env_config.define(ConfigField.coerce(key, spec))

    def register_dynamic_config_field(self, field: "ConfigField") -> str:
        """Register a single ad-hoc ``ConfigField`` this container defines
        at call time (``get_config(ConfigField(...))``)."""
        self.env_config.define(field)
        return field.name

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
            return self.register_dynamic_config_field(key)
        return key

    def get_config(self, key: "ConfigKeyType", type: "ConfigType | None" = None, default: "Any" = MISSING) -> "T":
        return self.env_config.get(self._resolve_config_key(key), type=type, default=default)

    def get_config_later(self, key: "ConfigKeyType", type: "ConfigType | None" = None, default: "Any" = MISSING) -> "T":
        return lazy_load(self.env_config.get, self._resolve_config_key(key), type=type, default=default)

    def make_exec_context(self, commands: "str | Iterable[str]") -> "OperationContext":
        from .context import OperationContext

        containers = self.manager.installed_state.get(resolve=True)
        if self not in containers:
            raise ContainerError(f"{self} is not installed")

        context = OperationContext()
        context.actions = [commands] if isinstance(commands, str) else list(filter(None, commands))
        context.project_containers = containers
        context.target_containers = [self]
        context.is_full_project = False
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
        # manager.temp_path is already environ.get_temp_path("container") --
        # joining another "container" segment here double-nested it into
        # .../temp/container/container/<name>.
        path = utils.join_path(self.manager.temp_path, self.name, *paths)
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def choose_service(self, name: "str | None" = None) -> "dict[str, Any] | None":
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

    def get_docker_file_destination(self) -> "Path":
        """Pure path computation, no write -- where a rendered Dockerfile
        WOULD go, regardless of whether it has actually been written yet.
        Used by ``load_docker_compose`` (a read-only render, referenced by
        Plan/Doctor/config-list) to fill in a build service's
        ``dockerfile`` field without writing anything; the real write only
        ever happens via ``get_docker_file_path()``, itself only called
        during real compose file generation (``write_docker_compose_file``)
        for actual execution."""
        return _compose.docker_file_destination(self)

    def get_docker_context_path(self) -> "Path":
        return self.get_source_path()

    def get_service_name(self, key: str) -> str:
        return f"{self.project_name}-{key}"

    def is_depend_on(self, name: str) -> bool:
        next_items = set(self.dependencies)
        exclude_items = set()
        while next_items:
            if name in next_items:
                return True
            exclude_items.update(next_items)
            current_items = next_items
            next_items = set()
            for next_name in current_items:
                # A dependency's defining container can go missing (its repo
                # was removed while something else installed still names it
                # as a dependency); skip it rather than crash, so `remove`
                # stays usable as the way to recover from that state instead
                # of being blocked by it.
                next_container = self.containers.get(next_name)
                if next_container is None:
                    continue
                for next_dependency in next_container.dependencies:
                    if next_dependency not in exclude_items:
                        next_items.add(next_dependency)
        return False

    def render_template(self, source: "PathType", destination: "PathType | None" = None, **kwargs: "Any") -> str:
        return _template.render_template(self, source, destination=destination, **kwargs)


    def __repr__(self):
        return f"Container<{self.name}>"


class SourceContainer(BaseContainer):
    __abstract__ = True

    def __init__(self, manager: "ContainerManager", root_path: "PathType", name: str = None):
        super().__init__(manager, root_path, name=name)
        self.add_start_hook(("init_source_code", self.name), self._prepare_source,
                            name="init_source_code", order=50, source="builtin")

    @property
    def _source_url(self):
        raise NotImplementedError()

    @property
    def _source_path(self):
        raise NotImplementedError()

    def _handle_source_file(self, source: "PathType", destination: "PathType"):
        from linktools.utils import safe_extract
        safe_extract(source, destination)

    @property
    def _source_root(self) -> "Path":
        return self.get_app_path("source", get_md5(self._source_url))

    def get_docker_context_path(self) -> "Path":
        from pathlib import Path
        return self._source_root / "current" / Path(self._source_path)

    def _source_digest(self) -> "str | None":
        from pathlib import Path
        link = self._source_root / "current"
        if not link.is_symlink():
            return None
        value = Path(os.readlink(str(link)))
        if value.parent != Path("versions") or not re.fullmatch("[0-9a-f]{64}", value.name):
            raise ContainerError("Invalid source snapshot pointer: " + str(link))
        return value.name

    def get_build_revision(self, service: str) -> "str | None":
        import hashlib
        import json
        digest = self._source_digest()
        if digest is None:
            return None
        identity = json.dumps([str(self._source_url), str(self._source_path), digest],
                              ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def _prepare_source(self, context: "OperationContext") -> None:
        root = self._source_root
        refresh = bool(set(self.services).intersection(context.refresh_services))
        if self._source_digest() is not None and not refresh:
            return
        # Reuse an existing installation's validated downloaded archive.
        name = get_md5(self._source_url)
        legacy_archive = self.get_app_path("source", name + ".in")
        legacy_tree = self.get_app_path("source", name + ".out")
        if not refresh and legacy_archive.is_file() and legacy_tree.is_dir():
            import hashlib
            import shutil
            root.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(legacy_archive.read_bytes()).hexdigest()
            versions, archives = root / "versions", root / "archives"
            versions.mkdir(exist_ok=True)
            archives.mkdir(exist_ok=True)
            destination = versions / digest
            if not destination.exists():
                shutil.copytree(str(legacy_tree), str(destination))
            if not (archives / (digest + ".in")).exists():
                shutil.copy2(str(legacy_archive), str(archives / (digest + ".in")))
            self._publish_source(digest)
            return
        if not refresh:
            buildable = [spec for spec in self.services.values() if spec.get("build") is not None]
            if buildable and all(spec.get("image") and
                                 self.manager.image_preparer.image_exists(spec["image"])
                                 for spec in buildable):
                # Source bytes are only needed to build, never to reuse a local image.
                return
        self._fetch_source()

    def _publish_source(self, digest: str) -> None:
        import uuid
        root = self._source_root
        link = root / "current"
        temporary = root / (".current-" + uuid.uuid4().hex)
        try:
            temporary.symlink_to("versions/" + digest)
            os.replace(str(temporary), str(link))
        finally:
            if temporary.is_symlink():
                temporary.unlink()

    def _fetch_source(self, expected: "str | None" = None) -> None:
        import hashlib
        import shutil
        import tempfile
        from pathlib import Path

        root = self._source_root
        root.mkdir(parents=True, exist_ok=True)
        versions, archives = root / "versions", root / "archives"
        versions.mkdir(exist_ok=True)
        archives.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".source-", dir=str(root)) as directory:
            archive = Path(self.manager.environ.get_url_file(self._source_url).save(
                directory, "source.in"))
            with open(archive, "rb") as stream:
                checksum = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    checksum.update(chunk)
                digest = checksum.hexdigest()
            if expected is not None and digest != expected:
                raise ContainerError("Source changed unexpectedly; use --pull to refresh " + self.name)
            destination = versions / digest
            if not destination.exists():
                temporary = Path(tempfile.mkdtemp(prefix=".extract-", dir=str(root)))
                try:
                    self._handle_source_file(str(archive), str(temporary))
                    source_path = Path(self._source_path)
                    if (source_path.is_absolute() or ".." in source_path.parts or
                            not (temporary / source_path).is_dir()):
                        raise ContainerError("Missing source build directory for " + self.name)
                    os.rename(str(temporary), str(destination))
                except BaseException:
                    shutil.rmtree(str(temporary))
                    raise
            retained = archives / (digest + ".in")
            if not retained.exists():
                shutil.copy2(str(archive), str(retained))
        self._publish_source(digest)

    def prepare_build_context(self) -> None:
        import shutil
        import tempfile
        from pathlib import Path
        digest = self._source_digest()
        if digest is None:
            raise ContainerError("Source snapshot was not prepared for " + self.name)
        root = self._source_root
        destination = root / "versions" / digest
        if destination.is_dir():
            return
        archive = root / "archives" / (digest + ".in")
        if not archive.is_file():
            # Recover only the known input, never silently advance a branch URL.
            self._fetch_source(expected=digest)
            return
        temporary = Path(tempfile.mkdtemp(prefix=".restore-", dir=str(root)))
        try:
            self._handle_source_file(str(archive), str(temporary))
            if not (temporary / self._source_path).is_dir():
                raise ContainerError("Missing cached source build directory for " + self.name)
            os.rename(str(temporary), str(destination))
        except BaseException:
            shutil.rmtree(str(temporary))
            raise

    def on_removed(self, context: "OperationContext") -> None:
        utils.remove_file(self.get_app_path("source"))


class SimpleContainer(BaseContainer):

    def __init__(self, manager: "ContainerManager", root_path: str):
        super().__init__(
            manager,
            root_path,
            name=os.path.basename(root_path)
        )
