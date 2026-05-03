#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : entry.py
@time    : 2022/12/18
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

import abc
import functools
import inspect
import logging
import os
import sys
import textwrap
import traceback
from argparse import ArgumentParser, Action, Namespace
from argparse import RawDescriptionHelpFormatter, SUPPRESS
from pkgutil import walk_packages
from types import ModuleType, GeneratorType
from typing import TYPE_CHECKING

from .argparse import BooleanOptionalAction, ArgParseComplete, ConfigAction, ConfigLoader
from .. import utils
from ..core import environ, BaseCapability, ConfigProperty
from ..decorator import cached_property
from ..metadata import __missing__
from ..rich import get_log_handler, init_logging, _is_rich_available
from ..types import Error

if TYPE_CHECKING:
    from argparse import FileType, HelpFormatter
    from collections.abc import Callable, Generator, Iterable
    from typing import Any, Literal
    from rich.tree import Tree
    from ..core import BaseEnviron
    from ..types import T

    ERROR_HANDLER = Literal["error", "ignore", "warn"] | Callable[[str, Exception], None]


class CommandError(Error):
    """Base exception for command-line failures."""
    pass


class SubCommandError(CommandError):
    """Raised when subcommand discovery or execution fails."""
    pass


class NotFoundSubCommand(SubCommandError):
    """Raised when no runnable subcommand can be resolved."""
    pass


class CommandParser(ArgumentParser):

    """ArgumentParser subclass that applies deferred config actions."""
    def __init__(self, *args, command: "BaseCommand" = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._command = command

    @property
    def command(self) -> "BaseCommand":
        """Command.

        Returns:
            BaseCommand: The property value.
        """
        return self._command

    def parse_known_args(self, args=None, namespace=None):
        """Parse args and resolve deferred config actions.

        Args:
            args: Arguments passed to the operation.
            namespace: Argparse namespace to update.

        Returns:
            Any: The operation result.
        """
        namespace, args = super().parse_known_args(args, namespace)
        for action in self._actions:
            if isinstance(action, ConfigAction):
                value = getattr(namespace, action.dest, None)
                if isinstance(value, ConfigLoader):
                    value(parser=self, action=action, namespace=namespace)
        return namespace, args


class _CommandInfo:
    id: str
    parent_id: str
    module: str
    command: "BaseCommand | None"
    command_name: str
    command_description: str
    order: str


def _iter_entry_points(group: str, *, onerror: "ERROR_HANDLER" = "error"):
    try:
        from importlib.metadata import entry_points
    except ImportError:
        from importlib_metadata import entry_points

    eps = entry_points()
    eps = eps.get(group, []) \
        if isinstance(eps, dict) \
        else eps.select(group=group)
    for ep in eps:
        try:
            yield ep.load()
        except Exception as e:
            if callable(onerror):
                onerror(ep.name, e)
            elif onerror == "error":
                raise e
            elif onerror == "warn":
                environ.logger.warning(
                    f"Ignore {ep.name}, caused by {e.__class__.__name__}: {e}",
                    exc_info=True if environ.debug else None
                )
            elif onerror == "ignore":
                pass


def iter_module_commands(root: "ModuleType", *, onerror: "ERROR_HANDLER" = "error") -> "Generator[_CommandInfo, Any, Any]":
    """Yield command descriptors discovered from a command module package.

    Args:
        root (ModuleType): The root value.
        onerror (ERROR_HANDLER): The onerror value.

    Returns:
        Generator[_CommandInfo, Any, Any]: The operation result.

    Raises:
        Exception: Propagates errors raised while completing the operation.
    """
    prefix = root.__name__ + "."
    for finder, name, is_package in walk_packages(path=root.__path__, prefix=prefix):
        try:
            module = utils.import_module(name, spec=finder.find_spec(name))
            info = _CommandInfo()
            if is_package:
                info.id = name[len(prefix):]
                info.parent_id = getattr(module, "__parent__", None) or name[len(prefix):name.rfind(".")]
                info.module = module.__name__
                info.command = None
                info.command_name = getattr(module, "__command__", None) or info.id[info.id.rfind(".") + 1:]
                info.command_description = getattr(module, "__description__", None) or ""
                info.order = getattr(module, "__order__", None) or info.command_name
                yield info
            elif hasattr(module, "command") and isinstance(module.command, BaseCommand):
                info.id = name[len(prefix):]
                info.parent_id = module.command.parent or name[len(prefix):name.rfind(".")]
                info.module = module.command.module
                info.command = module.command
                info.command_name = module.command.name
                info.command_description = module.command.description
                info.order = module.command.order
                yield info
        except Exception as e:
            if callable(onerror):
                onerror(name, e)
            elif onerror == "error":
                raise e
            elif onerror == "warn":
                environ.logger.warning(
                    f"Ignore {name}, caused by {e.__class__.__name__}: {e}",
                    exc_info=True if environ.debug else None
                )
            elif onerror == "ignore":
                pass


def iter_entry_point_commands(group: str, *, onerror: "ERROR_HANDLER" = "error") -> "Generator[_CommandInfo, Any, Any]":
    """Yield command descriptors discovered from entry points.

    Args:
        group (str): The group value.
        onerror (ERROR_HANDLER): The onerror value.

    Returns:
        Generator[_CommandInfo, Any, Any]: The operation result.
    """
    for obj in _iter_entry_points(group, onerror=onerror):
        if isinstance(obj, CommandMain):
            info = _CommandInfo()
            info.id = _join_id(obj.command.parent, obj.command.name)
            info.parent_id = obj.command.parent
            info.module = obj.command.module  # ep.module
            info.command = obj.command
            info.command_name = obj.command.name
            info.command_description = obj.command.description
            info.order = obj.command.order
            yield info
        elif isinstance(obj, ModuleType):
            yield from iter_module_commands(obj, onerror=onerror)


def iter_entry_points_capabilities(group: str, *, onerror: "ERROR_HANDLER" = "error"):
    """Yield capability objects discovered from entry points.

    Args:
        group (str): The group value.
        onerror (ERROR_HANDLER): The onerror value.

    Returns:
        Iterator[Any]: Generated values.
    """
    for obj in _iter_entry_points(group, onerror=onerror):
        if isinstance(obj, BaseCapability):
            yield obj
        elif inspect.isclass(obj) and issubclass(obj, BaseCapability):
            yield obj()


def _filter_kwargs(kwargs):
    return {k: v for k, v in kwargs.items() if v is not __missing__}


class _SubCommandActionInfo:

    def __init__(self, action: "Action", no_param: bool):
        self.action = action
        self.no_param = no_param

    @property
    def dest(self):
        return self.action.dest

    def __repr__(self):
        return f"SubCommandActionInfo(dest={self.dest})"


_subcommand_index: int = 0
_subcommand_map: "dict[str, set[str]]" = {}


class _SubCommandMethodInfo:

    def __init__(self):
        global _subcommand_index
        _subcommand_index += 1
        self.name = None
        self.order = None
        self.pass_args = False
        self.index = _subcommand_index
        self.kwargs: "dict[str, Any] | None" = None
        self.func: "Callable[..., int | None] | None" = None
        self.arguments: "list[_SubCommandMethodArgumentInfo]" = []

    def set_args(self, name: str, **kwargs: "Any"):
        self.name = name
        self.kwargs = _filter_kwargs(kwargs)
        return self

    def __repr__(self):
        return f"SubCommandMethod(func={self.func.__qualname__})"


class _SubCommandMethodArgumentInfo:

    def __init__(self):
        self.args: "tuple[str] | None" = None
        self.kwargs: "dict[str, Any] | None" = None
        self.action: "str | type[Action] | None" = None

    def set_args(self, *args: str, **kwargs: "Any"):
        self.args = args
        self.kwargs = _filter_kwargs(kwargs)
        return self


def subcommand(
        name: str,
        *,
        help: str = __missing__,
        aliases: "list[str]" = __missing__,
        prog: str = __missing__,
        usage: str = __missing__,
        description: str = __missing__,
        epilog: str = __missing__,
        parents: "list[ArgumentParser]" = __missing__,
        formatter_class: "type[HelpFormatter]" = __missing__,
        prefix_chars: str = __missing__,
        fromfile_prefix_chars: str = __missing__,
        argument_default: "Any" = __missing__,
        conflict_handler: str = __missing__,
        add_help: bool = __missing__,
        allow_abbrev: bool = __missing__,
        pass_args: bool = False,
        order: str = None):
    """Subcommand.

    Args:
        name (str): Name to resolve.
        help (str): The help value.
        aliases (List[str]): The aliases value.
        prog (str): The prog value.
        usage (str): The usage value.
        description (str): The description value.
        epilog (str): The epilog value.
        parents (List[ArgumentParser]): The parents value.
        formatter_class (Type[HelpFormatter]): The formatter_class value.
        prefix_chars (str): The prefix_chars value.
        fromfile_prefix_chars (str): The fromfile_prefix_chars value.
        argument_default (Any): The argument_default value.
        conflict_handler (str): The conflict_handler value.
        add_help (bool): The add_help value.
        allow_abbrev (bool): The allow_abbrev value.
        pass_args (bool): The pass_args value.
        order (str): The order value.

    Returns:
        Any: The operation result.

    Raises:
        Exception: Propagates errors raised while completing the operation.
    """

    def decorator(func):
        if not hasattr(func, "__subcommand_info__"):
            setattr(func, "__subcommand_info__", _SubCommandMethodInfo())

        subcommand_info = func.__subcommand_info__
        subcommand_info.func = func
        subcommand_info.pass_args = pass_args
        subcommand_info.order = order
        subcommand_info.set_args(
            name,
            help=help if help is not __missing__ else "",
            aliases=aliases,
            prog=prog,
            usage=usage,
            description=description,
            epilog=epilog,
            parents=parents,
            formatter_class=formatter_class,
            prefix_chars=prefix_chars,
            fromfile_prefix_chars=fromfile_prefix_chars,
            argument_default=argument_default,
            conflict_handler=conflict_handler,
            add_help=add_help,
            allow_abbrev=allow_abbrev
        )

        index = func.__qualname__.rfind(".")
        if index < 0:
            raise SubCommandError(
                f"subcommand decorator must be used in class method, "
                f"but {func.__qualname__} is not")

        class_name = f"{func.__module__}.{func.__qualname__[:index]}"
        func_name = func.__qualname__[index + 1:]

        _subcommand_map.setdefault(class_name, set())
        if func_name in _subcommand_map[class_name]:
            raise SubCommandError(
                f"Redeclared subcommand method '{func.__qualname__}' defined")
        _subcommand_map[class_name].add(func_name)

        return func

    return decorator


def subcommand_argument(
        name_or_flag: str,
        *name_or_flags: str,
        no_param: bool = False,
        action: "str | type[Action]" = __missing__,
        choices: "Iterable[T]" = __missing__,
        const: "Any" = __missing__,
        default: "Any" = __missing__,
        dest: str = __missing__,
        help: str = __missing__,
        metavar: "str | tuple[str, ...]" = __missing__,
        nargs: "int | str" = __missing__,
        required: bool = __missing__,
        type: "type[int | float | str] | Callable[[str], T] | FileType" = __missing__,
        **kwargs: "Any"):
    """Subcommand argument.

    Args:
        name_or_flag (str): The name_or_flag value.
        name_or_flags (str): The name_or_flags value.
        no_param (bool): The no_param value.
        action (Union[str, Type[Action]]): Argparse action being resolved.
        choices (Iterable[T]): The choices value.
        const (Any): The const value.
        default (Any): Value returned when no explicit value is available.
        dest (str): The dest value.
        help (str): The help value.
        metavar (Union[str, Tuple[str, ...]]): The metavar value.
        nargs (Union[int, str]): The nargs value.
        required (bool): The required value.
        type (Union[Type[Union[int, float, str]], Callable[[str], T], FileType]): Target type used to cast the value.
        kwargs (Any): Keyword arguments passed to the operation.

    Returns:
        Any: The operation result.
    """

    def decorator(func):
        subcommand_argument_info = _SubCommandMethodArgumentInfo()
        subcommand_argument_info.set_args(
            *[name_or_flag, *name_or_flags],
            no_param=no_param,
            action=action,
            nargs=nargs,
            const=const,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar,
            dest=dest,
            **kwargs
        )

        if not hasattr(func, "__subcommand_info__"):
            setattr(func, "__subcommand_info__", _SubCommandMethodInfo())

        subcommand_info = func.__subcommand_info__
        subcommand_info.arguments.append(subcommand_argument_info)

        return func

    return decorator


class _SubCommandInfo:
    node: "SubCommand"
    children: "list[_SubCommandInfo]"

    def __init__(self, subcommand: "SubCommand | _SubCommandInfo"):
        self.node = subcommand.node if isinstance(subcommand, _SubCommandInfo) else subcommand
        self.children = []

    def __repr__(self):
        return f"SubCommandInfo(node={self.node.id})"


def _join_id(*ids: str):
    return "#".join([id for id in ids if id])


class SubCommand(metaclass=abc.ABCMeta):
    """SubCommand."""

    ROOT_ID = _join_id()

    def __init__(self, name: str, description: str, id: str = None, parent_id: str = None, order: str = None):
        self.id = id or _join_id(parent_id, name)
        self.parent_id = parent_id or self.ROOT_ID
        self.name = name
        self.description = description
        self.order = order or self.name

    @property
    def has_parent(self):
        """Has parent.

        Returns:
            Any: The property value.
        """
        return self.parent_id != self.ROOT_ID

    @property
    def is_group(self):
        """Return whether group is true.

        Returns:
            Any: The property value.
        """
        return False

    def create_parser(self, type: "Callable[..., CommandParser]") -> "CommandParser":
        """Create a command parser.

        Args:
            type (Callable[..., CommandParser]): Target type used to cast the value.

        Returns:
            CommandParser: The operation result.
        """
        return type(self.name, help=self.description)

    @abc.abstractmethod
    def run(self, args: "Namespace"):
        """Run.

        Args:
            args (Namespace): Arguments passed to the operation.
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(id='{self.id}', parent_id='{self.parent_id}', name='{self.name}')"


class SubCommandGroup(SubCommand):
    """Subcommand placeholder that groups child commands."""

    @property
    def is_group(self):
        """Return whether group is true.

        Returns:
            Any: The property value.
        """
        return True

    def create_parser(self, type: "Callable[..., CommandParser]") -> "CommandParser":
        """Create a parser for a subcommand group.

        Args:
            type (Callable[..., CommandParser]): Target type used to cast the value.

        Returns:
            CommandParser: The operation result.
        """
        parser = type(self.name, help=self.description)
        parser.set_defaults(**{f"__subcommand_help_{id(self):x}__": parser.print_help})
        return parser

    def run(self, args: "Namespace"):
        """Print help for this subcommand group.

        Args:
            args (Namespace): Arguments passed to the operation.

        Returns:
            Any: The operation result.
        """
        attr_name = f"__subcommand_help_{id(self):x}__"
        assert hasattr(args, attr_name)
        func = getattr(args, attr_name)
        return func()


class _SubCommandMethod(SubCommand):

    def __init__(self, info: "_SubCommandMethodInfo", target: "Any",
                 id: str = None, parent_id: str = None,
                 order: str = None):
        super().__init__(
            id=id,
            parent_id=parent_id,
            name=info.name,
            description=info.kwargs.get("description", None) or info.kwargs.get("help", None) or "",
            order=order,
        )
        self.info = info
        self.target = target

    def create_parser(self, type: "Callable[..., CommandParser]") -> "CommandParser":

        actions = []
        method = getattr(self.target, self.info.func.__name__)
        parser = type(self.name, **self.info.kwargs)
        parser.set_defaults(**{f"__subcommand_actions_{id(self):x}__": actions})

        for argument in reversed(self.info.arguments):
            argument_args = argument.args
            argument_kwargs = dict(argument.kwargs)

            no_param = argument_kwargs.pop("no_param", __missing__)

            # Resolve dest so annotated arguments match method parameters.
            dest = argument_kwargs.get("dest", __missing__)
            if dest is __missing__:
                prefix_chars = parser.prefix_chars
                if not argument_args or len(argument_args) == 1 and argument_args[0][0] not in prefix_chars:
                    dest = argument_args[0]
                    argument_kwargs["required"] = __missing__  # Positional arguments cannot set required here.
                else:
                    option_strings = []
                    long_option_strings = []
                    for option_string in argument_args:
                        option_strings.append(option_string)
                        if len(option_string) > 1 and option_string[1] in prefix_chars:
                            long_option_strings.append(option_string)
                    dest_option_string = long_option_strings[0] if long_option_strings else option_strings[0]
                    dest = dest_option_string.lstrip(prefix_chars)
                    if not dest:
                        raise SubCommandError(
                            f"Parse subcommand argument dest error, "
                            f"{self.info} argument `{', '.join(argument_args)}` require dest=...")
                    dest = dest.replace('-', '_')
                    argument_kwargs["dest"] = dest

            # Validate that dest exists in the method signature.
            signature = inspect.signature(method)
            if not no_param and dest and dest not in signature.parameters:
                raise SubCommandError(
                    f"Check subcommand parameter error, {self.info} has no `{dest}` parameter. {os.linesep}"
                    f"You can do any of the following: {os.linesep}"
                    f"1. add `{dest}` parameter to {self.info}, {os.linesep}"
                    f"2. add `no_param=True` parameter to argument `{', '.join(argument_args)}`.")

            # Fill defaults from method parameter annotations.
            parameter = signature.parameters[dest] if not no_param else None
            if "config" in argument_kwargs:
                config = argument_kwargs.get("config")
                if isinstance(config, ConfigProperty):
                    if "action" not in argument_kwargs:
                        argument_kwargs.setdefault("action", ConfigAction)
                        if parameter and parameter.annotation != signature.empty:
                            if parameter.annotation in (int, float, str, bool):
                                argument_kwargs.setdefault("type", parameter.annotation)
                    argument_kwargs.setdefault("required", False)

            if parameter and "default" not in argument_kwargs:
                if parameter.default != signature.empty:
                    argument_kwargs.setdefault("default", parameter.default)
                    argument_kwargs.setdefault("required", False)
                else:
                    argument_kwargs.setdefault("required", True)

            if parameter and "action" not in argument_kwargs:
                if parameter.annotation != signature.empty:
                    if parameter.annotation in (int, float, str):
                        argument_kwargs.setdefault("type", parameter.annotation)
                    elif parameter.annotation == bool:
                        if argument_kwargs.get("default", False):
                            argument_kwargs.setdefault("action", "store_false")
                        else:
                            argument_kwargs.setdefault("action", "store_true")

            action = parser.add_argument(*argument_args, **_filter_kwargs(argument_kwargs))
            actions.append(_SubCommandActionInfo(action, no_param=no_param))

        return parser

    def run(self, args: "Namespace"):
        method = getattr(self.target, self.info.func.__name__)

        attr_name = f"__subcommand_actions_{id(self):x}__"
        assert hasattr(args, attr_name)
        actions = getattr(args, attr_name)

        method_args = []
        if self.info.pass_args:
            method_args.append(args)

        method_kwargs = dict()
        for action in actions:
            if not action.no_param:
                method_kwargs[action.dest] = getattr(args, action.dest)

        return method(*method_args, **method_kwargs)


class SubCommandWrapper(SubCommand):

    """Wrap a BaseCommand so it can be used as a subcommand."""
    def __init__(self, command: "BaseCommand",
                 id: str = None, parent_id: str = None,
                 name: str = None, description: str = None,
                 order: str = None):
        super().__init__(
            id=id or _join_id(command.parent, command.name),
            parent_id=parent_id or _join_id(command.parent),
            name=name or command.name,
            description=description or command.description,
            order=order or command.order,
        )
        self.command = command

    def create_parser(self, type: "Callable[..., CommandParser]") -> "CommandParser":
        """Create a parser for the wrapped command.

        Args:
            type (Callable[..., CommandParser]): Target type used to cast the value.

        Returns:
            CommandParser: The operation result.
        """
        return self.command.create_parser(self.name, help=self.description, type=type)

    def run(self, args: "Namespace"):
        """Run the wrapped command.

        Args:
            args (Namespace): Arguments passed to the operation.

        Returns:
            Any: The operation result.
        """
        return self.command(args)


class SubCommandMixin:

    """Mixin that discovers, registers, and prints subcommands."""
    def walk_subcommands(self: "BaseCommand", target: "Any", parent_id: str = None) -> "Generator[SubCommand, None, None]":
        """Yield subcommands discovered from a target object.

        Args:
            target (Any): The target value.
            parent_id (str): The parent_id value.

        Returns:
            Generator[SubCommand, None, None]: The operation result.
        """

        if isinstance(target, SubCommand):
            yield target

        elif isinstance(target, (list, tuple, set, GeneratorType)):
            for item in target:
                yield from self.walk_subcommands(item, parent_id=parent_id)

        elif isinstance(target, _CommandInfo):
            if target.command:
                yield SubCommandWrapper(
                    target.command,
                    id=_join_id(parent_id, target.id),
                    parent_id=_join_id(parent_id, target.parent_id),
                    order=target.order,
                )
            else:
                yield SubCommandGroup(
                    target.command_name, target.command_description,
                    id=_join_id(parent_id, target.id),
                    parent_id=_join_id(parent_id, target.parent_id),
                    order=target.order,
                )

        elif isinstance(target, ModuleType):
            for c in iter_module_commands(target, onerror="warn"):
                if c.command:
                    yield SubCommandWrapper(
                        c.command,
                        id=_join_id(parent_id, c.id),
                        parent_id=_join_id(parent_id, c.parent_id),
                        order=c.order,
                    )
                else:
                    yield SubCommandGroup(
                        c.command_name, c.command_description,
                        id=_join_id(parent_id, c.id),
                        parent_id=_join_id(parent_id, c.parent_id),
                        order=c.order,
                    )

        else:
            subcommand_map: "dict[str, list[_SubCommandMethod]]" = {}
            for clazz in target.__class__.mro():
                class_name = f"{clazz.__module__}.{clazz.__qualname__}"
                if class_name not in _subcommand_map:
                    continue
                for func_name in _subcommand_map[class_name]:
                    if not hasattr(clazz, func_name):
                        continue
                    func = getattr(clazz, func_name)
                    if not hasattr(func, "__subcommand_info__"):
                        continue
                    info: "_SubCommandMethodInfo" = func.__subcommand_info__
                    subcommand = _SubCommandMethod(info, target, parent_id=parent_id, order=info.order)
                    subcommand_map.setdefault(subcommand.name, list())
                    subcommand_map[info.name].append(subcommand)

            command_infos: "list[tuple[int, _SubCommandMethod]]" = []
            for name, subcommands in subcommand_map.items():
                command_infos.append((min([c.info.index for c in subcommands]), subcommands[0]))
            for _, subcommand in sorted(command_infos, key=lambda o: o[0]):
                yield subcommand

    def add_subcommands(
            self: "BaseCommand",
            parser: "CommandParser" = None,
            target: "Any" = None,
            required: bool = False,
            sort: bool = False,
    ) -> "list[_SubCommandInfo]":
        """Register discovered subcommands on a parser.

        Args:
            parser (CommandParser): Argument parser to configure or inspect.
            target (Any): The target value.
            required (bool): The required value.
            sort (bool): The sort value.

        Returns:
            List[_SubCommandInfo]: The operation result.

        Raises:
            Exception: Propagates errors raised while completing the operation.
        """
        target = target or self
        target_parser = parser or self._argument_parser

        subcommand_list = tuple(self.walk_subcommands(target))
        subcommand_maps = {subcommand_list[i].id: (i, subcommand_list[i]) for i in range(len(subcommand_list))}
        subcommand_index = {}
        for i in range(len(subcommand_list)):
            temp_subcommand = subcommand = subcommand_list[i]
            group = [subcommand.order if sort else i]
            while temp_subcommand.has_parent:
                if temp_subcommand.parent_id not in subcommand_maps:
                    raise SubCommandError(f"{temp_subcommand} has no parent subparser")
                parent_index, parent_subcommand = subcommand_maps.get(temp_subcommand.parent_id)
                if parent_subcommand is None or not parent_subcommand.is_group:
                    raise SubCommandError(f"{temp_subcommand} has no parent subparser")
                group.append(parent_subcommand.order if sort else parent_index)
                temp_subcommand = parent_subcommand
            subcommand_index[subcommand.id] = tuple(reversed(group))

        parsers = {}
        root_parser = parser.add_subparsers(metavar="COMMAND", help="Command Help")
        root_parser.required = required
        subcommand_infos: "list[_SubCommandInfo]" = sorted(
            [_SubCommandInfo(subcommand) for subcommand in subcommand_list],
            key=lambda x: subcommand_index.get(x.node.id)
        )
        for subcommand_info in subcommand_infos:
            subcommand = subcommand_info.node

            parent_parser = root_parser
            if subcommand.has_parent:
                parent_parser = parsers.get(subcommand.parent_id, None)
                if not parent_parser:
                    raise SubCommandError(f"{subcommand} has no parent subparser")

            parser = subcommand.create_parser(type=functools.partial(parent_parser.add_parser, command=self))
            parser.set_defaults(**{f"__subcommand_{id(self):x}__": subcommand})
            self.init_global_arguments(parser)

            if subcommand.is_group:
                subparser = parser.add_subparsers(metavar="COMMAND", help="Command Help")
                subparser.required = False
                parsers[subcommand.id] = subparser

            # Handle BaseCommand separately because init_arguments may add subcommands.
            if isinstance(subcommand, SubCommandWrapper):
                sub_subcommand_infos = parser.get_default(f"__subcommands_{id(subcommand.command):x}__")
                if sub_subcommand_infos:
                    subcommand_info.children.extend(
                        sub_subcommand_infos
                    )

        target_parser.set_defaults(**{f"__subcommands_{id(self):x}__": subcommand_infos})

        return subcommand_infos

    def parse_subcommand(self: "BaseCommand", args: "Namespace") -> "SubCommand | None":
        """Return the selected subcommand from parsed args.

        Args:
            args (Namespace): Arguments passed to the operation.

        Returns:
            Optional[SubCommand]: The operation result.
        """
        name = f"__subcommand_{id(self):x}__"
        if hasattr(args, name):
            subcommand = getattr(args, name)
            if isinstance(subcommand, SubCommand):
                return subcommand

        return None

    def run_subcommand(self: "BaseCommand", args: "Namespace") -> "int | None":
        """Run the selected subcommand from parsed args.

        Args:
            args (Namespace): Arguments passed to the operation.

        Returns:
            Optional[int]: The operation result.

        Raises:
            Exception: Propagates errors raised while completing the operation.
        """
        subcommand = self.parse_subcommand(args)
        if subcommand:
            return subcommand.run(args)
        raise NotFoundSubCommand("Not found subcommand")

    def print_subcommands(
            self: "BaseCommand",
            args: "Namespace",
            root: "SubCommand" = None,
            max_level: int = None
    ) -> None:
        """Print the registered subcommand tree.

        Args:
            args (Namespace): Arguments passed to the operation.
            root (SubCommand): The root value.
            max_level (int): The max_level value.

        Raises:
            Exception: Propagates errors raised while completing the operation.
        """
        name = f"__subcommands_{id(self):x}__"
        if not hasattr(args, name):
            raise SubCommandError("No subcommand has been added yet")

        root_id = SubCommand.ROOT_ID
        description = "All commands"
        if root:
            root_id = root.id
            if root.description:
                description = root.description
        elif self.description:
            description = self.description

        if _is_rich_available():
            from rich import get_console
            from rich.tree import Tree

            tree = self._make_subcommand_tree(
                Tree(f"📎 {description}"),
                getattr(args, name),
                root_id,
                max_level,
            )

            console = get_console()
            if self.environ.description != NotImplemented:
                console.print(self.environ.description, highlight=False)
            console.print(tree, highlight=False)
        else:
            if self.environ.description != NotImplemented:
                print(self.environ.description)
            print(description)
            self._print_subcommand_tree(getattr(args, name), root_id, max_level=max_level)

    def _print_subcommand_tree(
            self: "BaseCommand",
            infos: "list[_SubCommandInfo]",
            root_id: str,
            max_level: "int | None",
            level: int = 0,
    ) -> None:
        """
        Print a plain-text command tree when rich is unavailable.
        """
        for info in infos:
            if info.node.parent_id != root_id:
                continue

            current_level = level + 1
            if max_level is not None and current_level > max_level:
                continue

            prefix = "  " * level
            marker = "*" if info.node.is_group or info.children else "-"
            text = f"{prefix}{marker} {info.node.name}"
            if info.node.description:
                text = f"{text}: {info.node.description}"
            print(text)

            child_infos = info.children or infos
            child_root_id = SubCommand.ROOT_ID if info.children else info.node.id
            self._print_subcommand_tree(
                child_infos,
                child_root_id,
                max_level=max_level,
                level=current_level,
            )

    def _make_subcommand_tree(
            self: "BaseCommand",
            tree: "Tree",
            infos: "list[_SubCommandInfo]",
            root_id: str,
            max_level: "int | None"
    ) -> "Tree":
        nodes: "dict[str, tuple[Tree, int]]" = {}
        for info in infos:
            if info.node.parent_id == root_id:
                parent_node, parent_node_level = tree, 0
            elif info.node.parent_id in nodes:
                parent_node, parent_node_level = nodes.get(info.node.parent_id)
            else:
                self.logger.debug(f"Not found parent node id `{info.node.parent_id}`, skip")
                continue

            current_node_level = parent_node_level + 1
            current_node_expanded = max_level is None or max_level > current_node_level

            dbg_msg = f" [dim](group={info.node.is_group}, id={info.node.id}, order={info.node.order})[/dim]" \
                if self.environ.debug \
                else ""

            if info.node.is_group or info.children:
                logo = "📖" if current_node_expanded else "📘"
                text = f"{logo} [underline red]{info.node.name}[/underline red]{dbg_msg}"
                if info.node.description:
                    text = f"{text}: {info.node.description}"
                current_node = parent_node.add(text, expanded=current_node_expanded)
                nodes[info.node.id] = current_node, current_node_level
            else:
                text = f"👉 [bold red]{info.node.name}[/bold red]{dbg_msg}"
                if info.node.description:
                    text = f"{text}: {info.node.description}"
                current_node = parent_node.add(text, expanded=current_node_expanded)
                nodes[info.node.id] = current_node, current_node_level

            if info.children:
                current_max_level = max_level - current_node_level if max_level is not None else None
                self._make_subcommand_tree(
                    current_node,
                    info.children,
                    SubCommand.ROOT_ID,
                    current_max_level
                )

        return tree


class BaseCommand(SubCommandMixin, metaclass=abc.ABCMeta):
    """Base class for executable command-line commands."""

    @property
    def module(self) -> str:
        """Module.

        Returns:
            str: The property value.
        """
        return self.__module__

    @property
    def name(self) -> str:
        """Return the name.

        Returns:
            str: The property value.
        """
        name = self.module
        index = name.rfind(".")
        if index >= 0:
            name = name[index + 1:]
        return name

    @property
    def parent(self) -> "str | None":
        """Return the parent command id.

        Returns:
            Optional[str]: The property value.
        """
        return None

    @property
    def environ(self) -> "BaseEnviron":
        """Return the command environment.

        Returns:
            BaseEnviron: The property value.
        """
        return environ

    @property
    def config(self) -> "dict":
        """Return the configuration object.

        Returns:
            dict: The property value.
        """
        return self.environ.config

    @property
    def logger(self) -> "logging.Logger":
        """Return the root logger.

        Returns:
            logging.Logger: The property value.
        """
        return self.environ.logger

    @cached_property
    def description(self) -> str:
        """Description.

        Returns:
            str: The operation result.
        """
        return textwrap.dedent((self.__doc__ or "")).strip()

    @cached_property
    def order(self) -> str:
        """Order.

        Returns:
            str: The operation result.
        """
        return self.name

    @property
    def known_errors(self) -> "list[type[BaseException]]":
        """Return command-specific known error types.

        Returns:
            List[Type[BaseException]]: The property value.
        """
        return []

    @abc.abstractmethod
    def init_arguments(self, parser: "CommandParser") -> None:
        """Initialize parser arguments.

        Args:
            parser (CommandParser): Argument parser to configure or inspect.
        """
        pass

    @abc.abstractmethod
    def run(self, args: "Namespace") -> "int | None":
        """Run.

        Args:
            args (Namespace): Arguments passed to the operation.

        Returns:
            Optional[int]: The operation result.
        """
        pass

    def create_parser(
            self,
            *args: "Any",
            type: "Callable[..., CommandParser]" = CommandParser,
            formatter_class: "type[HelpFormatter]" = RawDescriptionHelpFormatter,
            conflict_handler="resolve",
            **kwargs: "Any"
    ) -> "CommandParser":
        """Create a command parser.

        Args:
            args (Any): Arguments passed to the operation.
            type (Callable[..., CommandParser]): Target type used to cast the value.
            formatter_class (Type[HelpFormatter]): The formatter_class value.
            conflict_handler: The conflict_handler value.
            kwargs (Any): Keyword arguments passed to the operation.

        Returns:
            CommandParser: The operation result.
        """
        description = kwargs.pop("description", None)
        if not description:
            description = self.description.strip()
            if description and self.environ.description != NotImplemented:
                description += os.linesep + os.linesep
                description += self.environ.description
        parser = type(
            *args,
            command=self,
            description=description,
            formatter_class=formatter_class,
            conflict_handler=conflict_handler,
            **kwargs
        )
        self.init_base_arguments(parser)
        self.init_arguments(parser)
        return parser

    @cached_property(lock=True)
    def _argument_parser(self) -> "CommandParser":
        parser = self.create_parser()
        self.init_global_arguments(parser)
        return parser

    def init_base_arguments(self, parser: "CommandParser") -> None:
        """Initialize base parser arguments.

        Args:
            parser (CommandParser): Argument parser to configure or inspect.
        """
        pass

    def init_global_arguments(self, parser: "CommandParser") -> None:
        """Initialize global parser arguments.

        Args:
            parser (CommandParser): Argument parser to configure or inspect.
        """

        environ = self.environ
        prefix = parser.prefix_chars[0] if parser.prefix_chars else "-"

        class VerboseAction(Action):

            def __call__(self, parser, namespace, values, option_string=None):
                logging.root.setLevel(logging.DEBUG)

        class SilentAction(Action):

            def __call__(self, parser, namespace, values, option_string=None):
                logging.disable(logging.CRITICAL)

        class DebugAction(Action):

            def __call__(self, parser, namespace, values, option_string=None):
                environ.global_config["DEBUG"] = True
                environ.logger.setLevel(logging.DEBUG)

        class LogTimeAction(BooleanOptionalAction):

            def __call__(self, parser, namespace, values, option_string=None):
                if option_string in self.option_strings:
                    value = not option_string.startswith("--no-")
                    handler = get_log_handler()
                    if handler:
                        handler.show_time = value

        class LogLevelAction(BooleanOptionalAction):

            def __call__(self, parser, namespace, values, option_string=None):
                if option_string in self.option_strings:
                    value = not option_string.startswith("--no-")
                    handler = get_log_handler()
                    if handler:
                        handler.show_level = value

        group = parser.add_argument_group(title="log options")
        group.add_argument(f"{prefix}{prefix}verbose", action=VerboseAction, nargs=0, const=True, dest=SUPPRESS,
                           help="increase log verbosity")
        group.add_argument(f"{prefix}{prefix}silent", action=SilentAction, nargs=0, const=True, dest=SUPPRESS,
                           help="disable all log output")
        group.add_argument(f"{prefix}{prefix}debug", action=DebugAction, nargs=0, const=True, dest=SUPPRESS,
                           help=f"increase {self.environ.name}'s log verbosity, and enable debug mode")

        if get_log_handler():
            group.add_argument(f"{prefix}{prefix}time", action=LogTimeAction, dest=SUPPRESS,
                               help="show log time")
            group.add_argument(f"{prefix}{prefix}level", action=LogLevelAction, dest=SUPPRESS,
                               help="show log level")

        if self.environ.version != NotImplemented:
            parser.add_argument(
                f"{prefix}{prefix}version", action="version", version=self.environ.version
            )

    @property
    def main(self) -> "CommandMain":
        """Return the command entry point.

        Returns:
            CommandMain: The property value.
        """
        return CommandMain(self, show_log_level=True, show_log_time=False)

    def __call__(self, args: "list[str] | Namespace" = None) -> int:
        """Call.

        Args:
            args (Union[List[str], Namespace]): Arguments passed to the operation.

        Returns:
            int: The operation result.
        """
        try:
            if not isinstance(args, Namespace):
                parser = ArgParseComplete.autocomplete(self._argument_parser)
                args = parser.parse_args(args)

            exit_code = self.run(args) or 0

        except (CommandError, *self.known_errors) as e:
            exit_code = 1
            error_type, error_message = e.__class__.__name__, str(e).strip()
            self.logger.error(
                f"{error_type}: {error_message}" if error_message else error_type,
                exc_info=True if self.environ.debug else None,
            )

        return exit_code


class BaseCommandGroup(BaseCommand, metaclass=abc.ABCMeta):

    """Base command that dispatches to registered subcommands."""
    def init_subcommands(self) -> "Any":
        """Return the target used to discover subcommands.

        Returns:
            Any: The operation result.
        """
        return self

    def init_arguments(self, parser: "CommandParser") -> None:
        """Register subcommands on the parser.

        Args:
            parser (CommandParser): Argument parser to configure or inspect.
        """
        self.add_subcommands(
            parser=parser,
            target=self.init_subcommands(),
        )

    def run(self, args: "Namespace") -> "int | None":
        """Dispatch to the selected subcommand or print the command tree.

        Args:
            args (Namespace): Arguments passed to the operation.

        Returns:
            Optional[int]: The operation result.
        """
        subcommand = self.parse_subcommand(args)
        if not subcommand or subcommand.is_group:
            return self.print_subcommands(args, subcommand, max_level=2)
        return subcommand.run(args)


class CommandMain:

    """Callable command entry point with logging and error handling."""
    def __init__(
            self,
            command: "BaseCommand", *,
            show_log_time: bool = False,
            show_log_level: bool = False,
            expand_user: bool = True,
            exit_on_return: bool = False,
    ):
        self._command = command
        self.show_log_level = show_log_level
        self.show_log_time = show_log_time
        self.expand_user = expand_user
        self.exit_on_return = exit_on_return

    @property
    def command(self) -> "BaseCommand":
        """Command.

        Returns:
            BaseCommand: The property value.
        """
        return self._command

    def init_logging(self):
        """Initialize logging for command execution."""
        init_logging(
            level=logging.INFO,
            show_time=self.show_log_time,
            show_level=self.show_log_level,
        )

    def __call__(self, args: "list[str]" = None) -> int:
        """Call.

        Args:
            args (List[str]): Arguments passed to the operation.

        Returns:
            int: The operation result.
        """
        self.init_logging()

        try:
            if self.expand_user:
                args = sys.argv[1:] if args is None else args
                args = tuple(os.path.expanduser(arg) for arg in args)
            result = self.command(args=args)
        except SystemExit as e:
            result = e.code
        except (KeyboardInterrupt, EOFError) as e:
            error_type, error_message = e.__class__.__name__, str(e).strip()
            self.command.logger.error(
                f"{error_type}: {error_message}" if error_message else error_type,
                exc_info=True if self.command.environ.debug else None,
            )
            result = 130  # https://tldp.org/LDP/abs/html/exitcodes.html#EXITCODESREF
        except:
            if self.command.environ.debug and _is_rich_available():
                from rich import get_console
                get_console().print_exception(show_locals=True)
            else:
                self.command.logger.error(traceback.format_exc())
            result = 1

        if self.exit_on_return:
            sys.exit(result)

        return result
