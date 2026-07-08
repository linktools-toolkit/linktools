#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : tools.py
@time    : 2018/12/11
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
import pathlib
import shutil
import warnings
from collections import ChainMap
from typing import TYPE_CHECKING

from linktools import utils
from linktools.errors import ToolDefinitionError
from linktools.errors import ToolExecError, ToolNotFound, ToolNotSupport
from linktools.system import get_interpreter, get_interpreter_ident, get_shell_path, get_system
from linktools.runtime import popen
from linktools.decorator import cached_property, timeoutable
from linktools.types import MISSING

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator
    from typing import Any
    from ._environ import BaseEnviron
    from linktools.types import PathType, TimeoutType
    from linktools.runtime import Process

SUPPRESS = object()
VALIDATE_KEYS = set()
INTERNAL_KEYS = set()


def _parse_value(config: "ChainMap[str, Any]", key: str, default=None):
    value = utils.get_item(config, key, default=default)
    if not isinstance(value, dict):
        # not found "when", use config value
        return value

    # parse when block:
    # -----------------------------------------
    #  field:
    #  when: <== when_block
    #  - system: [darwin, linux]
    #  then: xxx
    #  - system: windows
    #  then: yyy
    #  - else: ~
    # -----------------------------------------
    when_block = utils.get_item(value, "when", default=SUPPRESS)
    if when_block is SUPPRESS or not isinstance(when_block, (tuple, list)):
        # not found "when", use default value
        return value

    for cond_block in when_block:
        # -----------------------------------------
        #  field:
        #  when:
        #  - system: [darwin, linux]
        #  then: xxx <== then_block
        #  - system: windows
        #  then: yyy
        #  - else: ~
        # -----------------------------------------
        value = utils.get_item(cond_block, "then", default=SUPPRESS)
        if value != SUPPRESS:
            for key in VALIDATE_KEYS:
                # parse validate block:
                # -----------------------------------------
                #  field:
                #  when:
                #  - system: [darwin, linux] <== validate_block
                #  then: xxx
                #  - system: windows
                #  then: yyy
                #  - else: ~
                # -----------------------------------------
                choice = utils.get_item(cond_block, key, default=SUPPRESS)
                if choice is not SUPPRESS:
                    if isinstance(choice, str):
                        if config[key] != choice:
                            break
                    elif isinstance(choice, (tuple, list, set)):
                        if config[key] not in choice:
                            break
            else:
                # all keys are verified, return "then"
                return value

        # -----------------------------------------
        #  field:
        #  when:
        #  - system: [darwin, linux]
        #  then: xxx
        #  - system: windows
        #  then: yyy
        #  - else: ~ <== else_block
        # -----------------------------------------
        value = utils.get_item(cond_block, "else", default=SUPPRESS)
        if value != SUPPRESS:
            # if it is an else block, return "else"
            return value

    # use default value
    return default  # ==> not found "else"


class ToolStub(object):

    """Lightweight descriptor for a tool definition."""
    def __init__(self, path: "PathType", name: str, environ: "BaseEnviron" = None):
        self.system = environ.system if environ else get_system()
        self.name = f"{name}.bat" if self.system == "windows" else name
        self.path = pathlib.Path(path, self.name)

    @property
    def exists(self) -> bool:
        """Return whether the tool executable exists.

        Returns:
            bool: The property value.
        """
        return self.path and os.path.exists(self.path)

    def create(self, cmdline: str) -> "PathType":
        """Create an executable stub for a command line.

        Args:
            cmdline (str): Command line string to write or execute.

        Returns:
            PathType: The operation result.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wt") as fd:
            if self.system == "windows":
                fd.write("@echo off\n")
                fd.write(f"{cmdline} %*\n")
            else:
                fd.write(f"#!{shutil.which('sh')}\n")
                fd.write(f"{cmdline} \"$@\"\n")
                # fd.write(f"trap 'kill 0' INT TERM\n")
                # fd.write(f"{cmdline} \"$@\" &\n")
                # fd.write(f"wait\n")
        os.chmod(self.path, 0o755)
        return self.path

    def remove(self):
        """Remove the executable stub if it exists."""
        utils.ignore_errors(os.remove, args=(self.path,))

    def __repr__(self):
        return f"ToolStub<{self.name}>"


class ToolProperty(object):

    """Descriptor that resolves a named tool from an environment."""
    def __init__(self, name=None, raw: bool = False, default: "Any" = None,
                 internal: bool = False, validate: bool = False):
        self.name = name
        self.raw = raw
        self.default = default
        self.internal = internal
        self.validate = validate


class ToolMeta(type):

    """Metaclass that binds ToolProperty descriptors."""
    def __new__(mcs, name, bases, attrs):
        attrs["__default__"] = default = {}
        for key in list(attrs.keys()):
            if isinstance(attrs[key], ToolProperty):
                prop: "ToolProperty" = attrs[key]
                prop_name = prop.name or key
                if prop.validate:
                    VALIDATE_KEYS.add(prop_name)
                if prop.internal:
                    INTERNAL_KEYS.add(prop_name)
                default[prop_name] = prop.default
                attrs[key] = mcs._make_property(prop, prop_name)
        return type.__new__(mcs, name, bases, attrs)

    @classmethod
    def _make_property(mcs, prop: "ToolProperty", name: str):
        return property(lambda self: self._raw_config.get(name)) \
            if prop.raw \
            else property(lambda self: self.config.get(name))


class Tool(metaclass=ToolMeta):
    """Executable tool wrapper with prepare and run helpers."""
    __default__: "dict"

    name: str = ToolProperty(default=MISSING, raw=True, internal=True)
    system: str = ToolProperty(default=MISSING, raw=True, internal=True, validate=True)
    machine: str = ToolProperty(default=MISSING, raw=True, internal=True, validate=True)
    version: str = ToolProperty(default="", raw=True)
    depends_on: "tuple" = ToolProperty(default=[], internal=True)
    download_url: str = ToolProperty(default=MISSING)
    target_path: str = ToolProperty(default=MISSING, internal=True)
    root_path: str = ToolProperty(default=MISSING, internal=True)
    unpack_path: str = ToolProperty(default=MISSING, internal=True)
    absolute_path: str = ToolProperty(default=MISSING, internal=True)
    cmdline: str = ToolProperty(default=MISSING)
    executable_cmdline: "tuple" = ToolProperty(default=[], internal=True)
    environment: "dict[str, str]" = ToolProperty(default={}, internal=True)

    def __init__(self, tools: "Tools", name: str, config: "dict[str, Any]", **kwargs):
        self._tools = tools
        self._config = config

        raw_config = ChainMap(config, self.__default__)
        new_config = dict(
            name=name,
            system=self._tools.environ.system,
            machine=self._tools.environ.machine
        )

        # set value from environment
        prefix = name.replace("-", "_")
        for key, value in raw_config.items():
            if key not in INTERNAL_KEYS:
                new_value = self._tools.config.get(f"{prefix}_{key}".upper(), default=None)
                if new_value is not None:
                    new_config[key] = new_value

        new_config.update(kwargs)
        self._raw_config = raw_config.new_child(new_config)

    @cached_property(lock=True)
    def config(self) -> "dict":
        """Config.

        Returns:
            dict: The operation result.
        """
        config = {
            key: _parse_value(self._raw_config, key)
            for key in self._raw_config
        }

        depends_on = utils.get_item(config, "depends_on") or []
        if not isinstance(depends_on, (str, tuple, list)):
            raise ToolDefinitionError(
                "%s depends_on type error: str/tuple/list expected, got %s" % (self, type(depends_on)))
        if isinstance(depends_on, str):
            depends_on = [depends_on]
        for dependency in depends_on:
            if dependency not in self._tools.all:
                raise ToolDefinitionError(
                    "%s.depends_on error: not found Tool<%s>" % (self, dependency))
        config["depends_on"] = depends_on

        # download url
        download_url = utils.get_item(config, "download_url") or ""
        if not isinstance(download_url, str):
            raise ToolDefinitionError(
                "%s download_url type error: str expected, got %s" % (self, type(download_url)))
        config["download_url"] = download_url.format(tools=self._tools, **config)

        unpack_path = utils.get_item(config, "unpack_path") or ""
        if not isinstance(unpack_path, str):
            raise ToolDefinitionError(
                "%s unpack_path type error: str expected, got %s" % (self, type(unpack_path)))

        target_path = utils.get_item(config, "target_path") or ""
        if not isinstance(target_path, str):
            raise ToolDefinitionError(
                "%s target_path type error: str expected, got %s" % (self, type(target_path)))

        absolute_path = utils.get_item(config, "absolute_path") or ""
        if not isinstance(absolute_path, str):
            raise ToolDefinitionError(
                "%s absolute_path type error: str expected, got %s" % (self, type(absolute_path)))

        if download_url and not unpack_path and not target_path:
            target_path = utils.guess_file_name(download_url)

        # target path: {target_path}
        # unpack path: {unpack_path}
        # layout: tools/<name>/versions/<version>-<platform>-<arch>/
        # root path: {data_path}/tools/{name}/versions/{version}-{platform}-{arch}/
        config["target_path"] = target_path = target_path.format(tools=self._tools, **config)
        config["unpack_path"] = unpack_path = unpack_path.format(tools=self._tools, **config)

        # multi-version path resolution.
        version_slug = self.version or "unknown"
        platform_arch = "%s-%s" % (self._tools.environ.system, self._tools.environ.machine)
        version_dir = "%s-%s" % (version_slug, platform_arch)
        config["root_path"] = root_path = str(self._tools.environ.get_data_path(
            "tools", self.name, "versions", version_dir))

        # Check if the active.json pointer redirects to a different version.
        active_path = self._tools.environ.get_data_path("tools", self.name, "active.json")
        if os.path.isfile(str(active_path)):
            import json
            try:
                with open(str(active_path)) as f:
                    active = json.load(f)
                active_ver = active.get("version", "")
                active_pa = active.get("platform_arch", "")
                if active_ver and active_pa:
                    alt_dir = "%s-%s" % (active_ver, active_pa)
                    alt_root = str(self._tools.environ.get_data_path(
                        "tools", self.name, "versions", alt_dir))
                    if os.path.isdir(alt_root):
                        root_path = alt_root
                        config["root_path"] = root_path
            except Exception:
                pass

        if absolute_path:
            config["absolute_path"] = absolute_path.format(tools=self._tools, **config)
        elif config["target_path"]:
            config["absolute_path"] = os.path.join(root_path, target_path)
        else:
            config["absolute_path"] = ""

        # set executable cmdline
        cmdline = utils.get_item(config, "cmdline")
        if cmdline in (MISSING, None):
            cmdline = config["name"]
        if not isinstance(cmdline, str):
            raise ToolDefinitionError(
                "%s cmdline type error: str expected, got %s" % (self, type(cmdline)))
        config["cmdline"] = cmdline

        if not utils.is_empty(cmdline):
            cmdline = shutil.which(cmdline)
            if not utils.is_empty(cmdline):
                try:
                    if os.path.samefile(cmdline, self._stub.path):
                        cmdline = ""
                except FileNotFoundError:
                    pass
        if not utils.is_empty(cmdline):
            config["absolute_path"] = cmdline
            config["executable_cmdline"] = [cmdline]
        else:
            executable_cmdline = utils.get_item(config, "executable_cmdline")
            if executable_cmdline:
                if not isinstance(executable_cmdline, (str, tuple, list)):
                    raise ToolDefinitionError(
                        "%s executable_cmdline type error: str/tuple/list expected, got %s" % (self, type(executable_cmdline)))
            else:
                # if executable_cmdline is empty,
                # set absolute_path as executable_cmdline
                executable_cmdline = config["absolute_path"]
            if isinstance(executable_cmdline, str):
                executable_cmdline = [executable_cmdline]
            config["executable_cmdline"] = [
                str(cmd).format(tools=self._tools, **config)
                for cmd in executable_cmdline
            ]

        return config

    @property
    def supported(self) -> bool:
        """Return whether the tool supports the current environment.

        Returns:
            bool: The property value.
        """
        if self.exists:
            return True
        if self.download_url:
            return True
        return False

    @property
    def exists(self) -> bool:
        """Return whether the tool executable exists.

        Returns:
            bool: The property value.
        """
        if self.absolute_path:
            if os.path.exists(self.absolute_path):
                return True
        return False

    @property
    def _stub(self) -> "ToolStub":
        return ToolStub(
            self._tools.stub_path,
            self.name,
            environ=self._tools.environ
        )

    def get(self, key: str, default: "Any" = None) -> "Any":
        """Get.

        Args:
            key (str): Configuration or item key.
            default (Any): Value returned when no explicit value is available.

        Returns:
            Any: The operation result.
        """
        value = self.config.get(key, default)
        if isinstance(value, str):
            value = value.format(tools=self._tools, **self.config)
        return value

    def copy(self, **kwargs) -> "Tool":
        """Create a copy with updated configuration.

        Args:
            kwargs: Keyword arguments passed to the operation.

        Returns:
            Tool: The operation result.
        """
        return Tool(self._tools, self.name, self._config, **kwargs)

    def prepare(self) -> None:
        """Prepare a tool for execution.

        Raises:
            Exception: Propagates errors raised while completing the operation.
        """
        if not self.supported:
            raise ToolNotSupport(
                f"{self} does not support on "
                f"{self._tools.environ.system} ({self._tools.environ.machine})")

        for dependency in self.depends_on:
            tool = self._tools[dependency]
            tool.prepare()

        # download and extract (manifest in staging BEFORE target)
        if not self.exists:
            # Tool-level lock (fix-plan §2.3.3): serializes concurrent
            # installs of the same tool across processes. Dependencies are
            # prepared above, outside this lock.
            with self._tools.environ.locks.process_lock("tool:" + self.name):
                if not self.exists:  # re-check under lock; another process may have installed it
                    self._tools.logger.info(f"Download {self}: {self.download_url}")
                    import uuid as _uuid
                    from .._download import DownloadRequest
                    temp_dir = self._tools.environ.get_temp_path("tools", "cache")
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    temp_path = str(temp_dir / utils.guess_file_name(self.download_url))
                    # DownloadManager owns atomic landing / resume / hash
                    # validation; call it directly (fix-plan §2.3.1). sha256/size
                    # are passed only when the tool definition declares them.
                    self._tools.environ.downloads.download(DownloadRequest(
                        url=self.download_url, destination=temp_path,
                        sha256=self.get("sha256", None) or None,
                        size=self.get("size", None) or None,
                    ))

                    # staging dir — everything happens here before atomic move.
                    staging = "%s.staging-%s" % (self.root_path, _uuid.uuid4().hex[:8])
                    os.makedirs(staging, exist_ok=True)
                    corrupt = None
                    try:
                        if not utils.is_empty(self.unpack_path):
                            self._tools.logger.debug(f"Extract {self} to {staging}")
                            utils.safe_extract(temp_path, staging)
                            os.remove(temp_path)
                        else:
                            target_in_staging = os.path.join(
                                staging,
                                os.path.relpath(self.absolute_path, self.root_path))
                            os.makedirs(os.path.dirname(target_in_staging) or staging, exist_ok=True)
                            shutil.move(temp_path, target_in_staging)

                        # write manifest INSIDE staging before move.
                        self._write_manifest(staging)

                        # Swap an existing (incomplete) root aside, then atomically
                        # put staging in its place. If the final move fails, restore
                        # the old dir so the tool stays usable (fix-plan §2.3.2).
                        if os.path.exists(self.root_path):
                            corrupt = self._make_corrupt_path(self.root_path)
                            os.replace(self.root_path, corrupt)
                        try:
                            os.replace(staging, self.root_path)
                            staging = None  # consumed
                        except BaseException:
                            if not os.path.exists(self.root_path) and corrupt \
                                    and os.path.exists(corrupt):
                                os.replace(corrupt, self.root_path)
                                corrupt = None
                            raise
                        if corrupt:
                            shutil.rmtree(corrupt, ignore_errors=True)
                    except BaseException:
                        if staging and os.path.exists(staging):
                            shutil.rmtree(staging, ignore_errors=True)
                        if corrupt and os.path.exists(corrupt):
                            shutil.rmtree(corrupt, ignore_errors=True)
                        raise

                    # Active pointer after successful activation.
                    self._set_active()

        if not os.access(self._stub.path, os.X_OK):
            self._tools.logger.debug(f"Create {self._stub}")
            self._stub.create(self.make_cmdline())

        # change tool file permission
        cmdline = self.executable_cmdline
        path = cmdline[0] if cmdline else ""
        if self.absolute_path == path and utils.is_sub_path(path, self.root_path):
            if not os.access(self.absolute_path, os.X_OK):
                self._tools.logger.debug(f"Chmod 755 {self.absolute_path}")
                os.chmod(self.absolute_path, 0o0755)

    def clear(self) -> None:
        """Clear cached or generated data (v2: removes version dir + active.json)."""
        if self._stub.exists:
            self._tools.logger.debug(f"Delete {self._stub}")
            self._stub.remove()
        if not self.exists:
            self._tools.logger.debug(f"{self} does not exist, skip")
            return
        # Remove the version dir.
        if self.root_path and os.path.isdir(self.root_path):
            self._tools.logger.debug(f"Delete {self.root_path}")
            shutil.rmtree(self.root_path, ignore_errors=True)
        # Remove the active.json pointer.
        active_path = str(self._tools.environ.get_data_path("tools", self.name, "active.json"))
        if os.path.isfile(active_path):
            utils.ignore_errors(os.remove, args=(active_path,))

    def popen(self, *args: "Any", **kwargs) -> "Process":
        """Start a process for this tool.

        Args:
            args (Any): Arguments passed to the operation.
            kwargs: Keyword arguments passed to the operation.

        Returns:
            Process: The operation result.
        """
        self.prepare()

        #  pass the tools-stub PATH via subprocess_env instead of
        # relying on the global os.environ mutation.
        env = self._tools.environ.subprocess_env()
        if self.environment:
            for key, value in self.environment.items():
                env.setdefault(key, value)
        kwargs.setdefault("default_env", env)

        # java or other
        executable_cmdline = self.executable_cmdline
        if executable_cmdline[0] in self._tools.all:
            args = [*executable_cmdline[1:], *args]
            tool = self._tools[executable_cmdline[0]]
            return tool.popen(*args, **kwargs)

        return popen(*[*executable_cmdline, *args], **kwargs)

    @timeoutable
    def exec(
            self,
            *args: "Any",
            timeout: "TimeoutType" = None,
            ignore_errors: bool = False,
            on_stdout: "Callable[[str], None]" = None,
            on_stderr: "Callable[[str], None]" = None,
            error_type: "Callable[[str], Exception]" = ToolExecError
    ) -> str:
        """Run a process command until completion.

        Args:
            args (Any): Arguments passed to the operation.
            timeout (TimeoutType): Maximum time to wait, or None to wait indefinitely.
            ignore_errors (bool): Whether command errors should be suppressed.
            on_stdout (Callable[[str], None]): Callback invoked for stdout output.
            on_stderr (Callable[[str], None]): Callback invoked for stderr output.
            error_type (Callable[[str], Exception]): Exception type raised for command failures.

        Returns:
            str: The operation result.
        """
        process = self.popen(*args, capture_output=True)
        return process.exec(
            timeout=timeout,
            ignore_errors=ignore_errors,
            on_stdout=on_stdout,
            on_stderr=on_stderr,
            error_type=error_type
        )

    def __repr__(self):
        return f"Tool<{self.name}>"

    def make_cmdline(self) -> str:
        """Return the command line used to invoke this tool through linktools.

        Returns:
            str: The operation result.
        """
        from ..cli import env
        return utils.list2cmdline([get_interpreter(), "-m", env.__name__, "tool", self.name])

    def _make_corrupt_path(self, root_path):
        """Return a unique path under the tools tree to quarantine a bad install.

        Lives next to the version dir (same filesystem, inside the tools tree)
        so os.replace stays atomic and a bad target never escapes the tools dir.
        """
        import uuid as _uuid
        corrupt_base = os.path.join(os.path.dirname(root_path), ".corrupt")
        os.makedirs(corrupt_base, exist_ok=True)
        return os.path.join(corrupt_base, "%s-%s" % (
            os.path.basename(root_path), _uuid.uuid4().hex[:8]))

    def _write_manifest(self, target_dir=None):
        """Write manifest.json (v4 §9.5: inside staging before move)."""
        import json
        target_dir = target_dir or self.root_path
        # entrypoint is stored relative to the install root so the same manifest
        # is valid whether read from staging or the final version dir (fix-plan §2.3.4).
        entrypoint = os.path.relpath(self.absolute_path, self.root_path) \
            if self.absolute_path else ""
        manifest = {
            "schema": 1,
            "name": self.name,
            "version": self.version,
            "platform": self._tools.environ.system,
            "architecture": self._tools.environ.machine,
            "source_url": self.download_url,
            "installed_at": _now_iso(),
            "entrypoint": entrypoint,
        }
        manifest_path = os.path.join(target_dir, "manifest.json")
        utils.atomic_write(manifest_path, json.dumps(manifest, indent=2))

    def _set_active(self):
        """Write active.json pointing at this version (v2 §8.3)."""
        import json
        active = {
            "version": self.version or "unknown",
            "platform_arch": "%s-%s" % (self._tools.environ.system, self._tools.environ.machine),
        }
        active_path = str(self._tools.environ.get_data_path("tools", self.name, "active.json"))
        utils.atomic_write(active_path, json.dumps(active, indent=2))


class Tools(object):

    """Registry and factory for tools available in an environment."""
    def __init__(self, environ: "BaseEnviron", config: "dict[str, Any]"):
        self.environ = environ
        self.logger = environ.get_logger("tools")
        self.config = environ.wrap_config(env_prefix="")
        self.all = self._parse_items(config)

    @cached_property
    def stub_path(self) -> "pathlib.Path":
        """Stub path.

        Returns:
            pathlib.Path: The operation result.
        """
        return self.environ.get_data_path(
            "scripts",
            get_interpreter_ident(),
            f"tools_v{self.environ.version}",
        )

    def keys(self) -> "Generator[str, None, None]":
        """Keys.

        Returns:
            Generator[str, None, None]: The operation result.
        """
        for k, v in self.all.items():
            if v.supported:
                yield k

    def values(self) -> "Generator[Tool, None, None]":
        """Values.

        Returns:
            Generator[Tool, None, None]: The operation result.
        """
        for k, v in self.all.items():
            if v.supported:
                yield v

    def items(self) -> "Generator[tuple[str, Tool], None, None]":
        """Items.

        Returns:
            Generator[Tuple[str, Tool], None, None]: The operation result.
        """
        for k, v in self.all.items():
            if v.supported:
                yield k, v

    def __iter__(self) -> "Iterator[Tool]":
        return iter([t for t in self.all.values() if t.supported])

    def __getitem__(self, item: str) -> "Tool":
        tool = self.all.get(item, None)
        if tool is None:
            raise ToolNotFound(f"Not found tool {item}")
        return tool

    def __getattr__(self, item: str) -> "Tool":
        return self[item]

    def __setitem__(self, key: str, value: "Tool"):
        self.all[key] = value

    def _parse_items(self, config: "dict[str, dict[str, Any]]") -> "dict[str, Tool]":
        result = {
            "shell": Tool(self, "shell", {
                "cmdline": None,
                "absolute_path": get_shell_path(),
            }),
            "python": Tool(self, "python", {
                "cmdline": None,
                "absolute_path": get_interpreter(),
            }),
        }

        for key, value in config.items():
            if not isinstance(value, dict):
                warnings.warn(f"dict was expected, got {type(value)}, ignored.")
                continue
            name = value.get("name", None)
            if name is None:
                if key.startswith("TOOL_"):
                    key = key[len("TOOL_"):]
                name = key.lower()
            result[name] = Tool(self, name, value)

        return result


def _now_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
