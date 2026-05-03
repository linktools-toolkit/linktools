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
from typing import TYPE_CHECKING, Tuple, List

from linktools import utils
from linktools.decorator import cached_property, timeoutable
from linktools.metadata import __missing__
from linktools.types import ToolNotFound, ToolNotSupport, ToolExecError

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator
    from typing import Any
    from ._environ import BaseEnviron
    from linktools.types import PathType, TimeoutType

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
    #   field:
    #     when:                                 <== when_block
    #       - system: [darwin, linux]
    #         then: xxx
    #       - system: windows
    #         then: yyy
    #       - else: ~
    # -----------------------------------------
    when_block = utils.get_item(value, "when", default=SUPPRESS)
    if when_block is SUPPRESS or not isinstance(when_block, (tuple, list)):
        # not found "when", use default value
        return value

    for cond_block in when_block:
        # -----------------------------------------
        #   field:
        #     when:
        #       - system: [darwin, linux]
        #         then: xxx                         <== then_block
        #       - system: windows
        #         then: yyy
        #       - else: ~
        # -----------------------------------------
        value = utils.get_item(cond_block, "then", default=SUPPRESS)
        if value != SUPPRESS:
            for key in VALIDATE_KEYS:
                # parse validate block:
                # -----------------------------------------
                #   field:
                #     when:
                #       - system: [darwin, linux]           <== validate_block
                #         then: xxx
                #       - system: windows
                #         then: yyy
                #       - else: ~
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
        #   field:
        #     when:
        #       - system: [darwin, linux]
        #         then: xxx
        #       - system: windows
        #         then: yyy
        #       - else: ~                           <== else_block
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
        self.system = environ.system if environ else utils.get_system()
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
                fd.write(f"@echo off\n")
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
    __default__: "Dict"

    name: str = ToolProperty(default=__missing__, raw=True, internal=True)
    system: str = ToolProperty(default=__missing__, raw=True, internal=True, validate=True)
    machine: str = ToolProperty(default=__missing__, raw=True, internal=True, validate=True)
    version: str = ToolProperty(default="", raw=True)
    depends_on: "tuple" = ToolProperty(default=[], internal=True)
    download_url: str = ToolProperty(default=__missing__)
    target_path: str = ToolProperty(default=__missing__, internal=True)
    root_path: str = ToolProperty(default=__missing__, internal=True)
    unpack_path: str = ToolProperty(default=__missing__, internal=True)
    absolute_path: str = ToolProperty(default=__missing__, internal=True)
    cmdline: str = ToolProperty(default=__missing__)
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
        assert isinstance(depends_on, (str, Tuple, List)), \
            f"{self} depends_on type error, " \
            f"str/tuple/list was expects, got {type(depends_on)}"
        if isinstance(depends_on, str):
            depends_on = [depends_on]
        for dependency in depends_on:
            assert dependency in self._tools.all, \
                f"{self}.depends_on error: not found Tool<{dependency}>"
        config["depends_on"] = depends_on

        # download url
        download_url = utils.get_item(config, "download_url") or ""
        assert isinstance(download_url, str), \
            f"{self} download_url type error, " \
            f"str was expects, got {type(download_url)}"
        config["download_url"] = download_url.format(tools=self._tools, **config)

        unpack_path = utils.get_item(config, "unpack_path") or ""
        assert isinstance(unpack_path, str), \
            f"{self} unpack_path type error, " \
            f"str was expects, got {type(unpack_path)}"

        target_path = utils.get_item(config, "target_path") or ""
        assert isinstance(target_path, str), \
            f"{self} target_path type error, " \
            f"str was expects, got {type(target_path)}"

        absolute_path = utils.get_item(config, "absolute_path") or ""
        assert isinstance(absolute_path, str), \
            f"{self} absolute_path type error, " \
            f"str was expects, got {type(absolute_path)}"

        if download_url and not unpack_path and not target_path:
            target_path = utils.guess_file_name(download_url)

        # target path: {target_path}
        # unpack path: {unpack_path}
        # root path: {data_path}/tools/{unpack_path}/
        # absolute path: {data_path}/tools/{unpack_path}/{target_path}
        config["target_path"] = target_path = target_path.format(tools=self._tools, **config)
        config["unpack_path"] = unpack_path = unpack_path.format(tools=self._tools, **config)
        paths = ["tools"]
        if not utils.is_empty(unpack_path):
            paths.append(unpack_path)
        else:
            paths.append(
                f"{self.name}-{self.version}"
                if self.version
                else self.name
            )
        config["root_path"] = root_path = str(self._tools.environ.get_data_path(*paths))

        if absolute_path:
            config["absolute_path"] = absolute_path.format(tools=self._tools, **config)
        elif config["target_path"]:
            config["absolute_path"] = os.path.join(root_path, target_path)
        else:
            config["absolute_path"] = ""

        # set executable cmdline
        cmdline = utils.get_item(config, "cmdline")
        if cmdline in (__missing__, None):
            cmdline = config["name"]
        assert isinstance(cmdline, str), \
            f"{self} cmdline type error, " \
            f"str was expects, got {type(cmdline)}"
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
                assert isinstance(executable_cmdline, (str, tuple, list)), \
                    f"{self} executable_cmdline type error, " \
                    f"str/tuple/list was expects, got {type(executable_cmdline)}"
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

        # download and unzip file
        if not self.exists:
            self._tools.logger.info(f"Download {self}: {self.download_url}")
            with self._tools.environ.get_url_file(self.download_url) as url_file:
                if not self.exists:
                    temp_dir = self._tools.environ.get_temp_path("tools", "cache")
                    temp_path = url_file.save(temp_dir)
                    if not utils.is_empty(self.unpack_path):
                        self._tools.logger.debug(f"Unpack {self} to {self.root_path}")
                        os.makedirs(self.root_path, exist_ok=True)
                        shutil.unpack_archive(temp_path, self.root_path)
                        os.remove(temp_path)
                    else:
                        self._tools.logger.debug(f"Move {self} to {self.absolute_path}")
                        os.makedirs(self.root_path, exist_ok=True)
                        shutil.move(temp_path, self.absolute_path)

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
        """Clear cached or generated data."""
        if self._stub.exists:
            self._tools.logger.debug(f"Delete {self._stub}")
            self._stub.remove()
        if not self.exists:
            self._tools.logger.debug(f"{self} does not exist, skip")
            return
        if not utils.is_empty(self.unpack_path):
            self._tools.logger.debug(f"Delete {self.root_path}")
            shutil.rmtree(self.root_path, ignore_errors=True)
        elif not utils.is_empty(self.root_path):
            self._tools.logger.debug(f"Delete {self.root_path}")
            shutil.rmtree(self.root_path, ignore_errors=True)

    def popen(self, *args: "Any", **kwargs) -> "utils.Process":
        """Start a process for this tool.

        Args:
            args (Any): Arguments passed to the operation.
            kwargs: Keyword arguments passed to the operation.

        Returns:
            utils.Process: The operation result.
        """
        self.prepare()

        if self.environment:
            env = kwargs.setdefault("default_env", {})
            for key, value in self.environment.items():
                env.setdefault(key, value)

        # java or other
        executable_cmdline = self.executable_cmdline
        if executable_cmdline[0] in self._tools.all:
            args = [*executable_cmdline[1:], *args]
            tool = self._tools[executable_cmdline[0]]
            return tool.popen(*args, **kwargs)

        return utils.popen(*[*executable_cmdline, *args], **kwargs)

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
        return utils.list2cmdline([utils.get_interpreter(), "-m", env.__name__, "tool", self.name])


class Tools(object):

    """Registry and factory for tools available in an environment."""
    def __init__(self, environ: "BaseEnviron", config: "dict[str, Dict]"):
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
            utils.get_interpreter_ident(),
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
                "absolute_path": utils.get_shell_path(),
            }),
            "python": Tool(self, "python", {
                "cmdline": None,
                "absolute_path": utils.get_interpreter(),
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
