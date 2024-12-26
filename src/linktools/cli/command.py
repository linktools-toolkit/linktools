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
import inspect
import logging
import os
import sys
import textwrap
import traceback
from argparse import ArgumentParser, Action, Namespace
from argparse import RawDescriptionHelpFormatter, SUPPRESS, FileType, HelpFormatter
from pkgutil import walk_packages
from types import ModuleType, GeneratorType
from typing import TYPE_CHECKING, Optional, Callable, List, Type, Tuple, Generator, Any, Iterable, Union, Set, Dict

from rich import get_console
from rich.tree import Tree

from .argparse import BooleanOptionalAction, auto_complete
from .. import utils
from .._environ import environ
from ..decorator import cached_property
from ..metadata import __missing__
from ..rich import LogHandler, init_logging
from ..types import Error

if TYPE_CHECKING:
    from typing import TypeVar, Union, Literal
    from .._environ import BaseEnviron

    T = TypeVar("T")
    ERROR_HANDLER = Union[Literal["error", "ignore", "warn"], Callable[[str, Exception], None]]


class CommandError(Error):
    pass


class SubCommandError(CommandError):
    pass


class _CommandInfo:
    id: str
    parent_id: str
    module: str
    command: "Optional[BaseCommand]"
    command_name: str
    command_description: str


def iter_module_commands(root: ModuleType, *, onerror: "ERROR_HANDLER" = "error") -> Generator[_CommandInfo, Any, Any]:
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
                yield info
            elif hasattr(module, "command") and isinstance(module.command, BaseCommand):
                info.id = name[len(prefix):]
                info.parent_id = module.command.parent or name[len(prefix):name.rfind(".")]
                info.module = module.command.module
                info.command = module.command
                info.command_name = module.command.name
                info.command_description = module.command.description
                yield info
        except Exception as e:
            if callable(onerror):
                onerror(name, e)
            elif onerror == "error":
                raise e
            elif onerror == "warn":
                environ.logger.warning(
                    f"Ignore {name}, caused by {e.__class__.__name__}: {e}",
                    exc_info=e if environ.debug else None
                )
            elif onerror == "ignore":
                pass


def iter_entry_point_commands(group: str, *, onerror: "ERROR_HANDLER" = "error") -> Generator[_CommandInfo, Any, Any]:
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
            main = ep.load()
            if isinstance(main, CommandMain):
                info = _CommandInfo()
                info.id = _join_id(main.command.parent, main.command.name)
                info.parent_id = main.command.parent
                info.module = main.command.module  # ep.module
                info.command = main.command
                info.command_name = main.command.name
                info.command_description = main.command.description
                yield info
        except Exception as e:
            if callable(onerror):
                onerror(ep.name, e)
            elif onerror == "error":
                raise e
            elif onerror == "warn":
                environ.logger.warning(
                    f"Ignore {ep.name}, caused by {e.__class__.__name__}: {e}",
                    exc_info=e if environ.debug else None
                )
            elif onerror == "ignore":
                pass


def _filter_kwargs(kwargs):
    return {k: v for k, v in kwargs.items() if v != __missing__}


_subcommand_index: int = 0
_subcommand_map: Dict[str, Set[str]] = {}


class _SubCommandMethodInfo:

    def __init__(self):
        global _subcommand_index
        _subcommand_index += 1
        self.name = None
        self.pass_args = False
        self.index = _subcommand_index
        self.kwargs: Optional[Dict[str, Any]] = None
        self.func: Optional[Callable[..., Optional[int]]] = None
        self.arguments: List[_SubCommandMethodArgumentInfo] = []

    def set_args(self, name: str, **kwargs: Any):
        self.name = name
        self.kwargs = _filter_kwargs(kwargs)
        return self

    def __repr__(self):
        return f"SubCommandMethod(func={self.func.__qualname__})"


class _SubCommandMethodArgumentInfo:

    def __init__(self):
        self.args: Optional[Tuple[str]] = None
        self.kwargs: Optional[Dict[str, Any]] = None
        self.action: Optional[Union[str, Type[Action]]] = None

    def set_args(self, *args: str, **kwargs: Any):
        self.args = args
        self.kwargs = _filter_kwargs(kwargs)
        return self


def subcommand(
        name: str,
        *,
        help: str = __missing__,
        aliases: List[str] = __missing__,
        prog: str = __missing__,
        usage: str = __missing__,
        description: str = __missing__,
        epilog: str = __missing__,
        parents: List[ArgumentParser] = __missing__,
        formatter_class: Type[HelpFormatter] = __missing__,
        prefix_chars: str = __missing__,
        fromfile_prefix_chars: str = __missing__,
        argument_default: Any = __missing__,
        conflict_handler: str = __missing__,
        add_help: bool = __missing__,
        allow_abbrev: bool = __missing__,
        pass_args: bool = False):
    """
    子命令装饰器
    """

    def decorator(func):
        if not hasattr(func, "__subcommand_info__"):
            setattr(func, "__subcommand_info__", _SubCommandMethodInfo())

        subcommand_info = func.__subcommand_info__
        subcommand_info.func = func
        subcommand_info.pass_args = pass_args
        subcommand_info.set_args(
            name,
            help=help if help != __missing__ else "",
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
        action: Union[str, Type[Action]] = __missing__,
        choices: "Iterable[T]" = __missing__,
        const: Any = __missing__,
        default: Any = __missing__,
        dest: str = __missing__,
        help: str = __missing__,
        metavar: Union[str, Tuple[str, ...]] = __missing__,
        nargs: Union[int, str] = __missing__,
        required: bool = __missing__,
        type: "Union[Type[Union[int, float, str]], Callable[[str], T], FileType]" = __missing__,
        **kwargs: Any):
    """
    子命令参数装饰器，与@subcommand配合使用
    """

    def decorator(func):
        subcommand_argument_info = _SubCommandMethodArgumentInfo()
        subcommand_argument_info.set_args(
            *[name_or_flag, *name_or_flags],
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

    def __init__(self, subcommand: "Union[SubCommand, _SubCommandInfo]"):
        self.node: SubCommand = subcommand.node if isinstance(subcommand, _SubCommandInfo) else subcommand
        self.children: List[_SubCommandInfo] = []

    def __repr__(self):
        return f"SubCommandInfo(node={self.node.id})"


def _join_id(*ids: str):
    return "#".join([id for id in ids if id])


class SubCommand(metaclass=abc.ABCMeta):
    """
    子命令接口
    """

    ROOT_ID = _join_id()

    def __init__(self, name: str, description: str, id: str = None, parent_id: str = None):
        self.id = id or _join_id(parent_id, name)
        self.parent_id = parent_id or self.ROOT_ID
        self.name = name
        self.description = description

    @property
    def has_parent(self):
        """
        是否有父命令
        """
        return self.parent_id != self.ROOT_ID

    @property
    def is_group(self):
        """
        是否是命令组
        """
        return False

    def create_parser(self, type: Callable[..., ArgumentParser]) -> ArgumentParser:
        """
        创建ArgumentParser对象
        """
        return type(self.name, help=self.description)

    @abc.abstractmethod
    def run(self, args: Namespace):
        """
        业务逻辑入口
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(id='{self.id}', parent_id='{self.parent_id}', name='{self.name}')"


class SubCommandGroup(SubCommand):

    @property
    def is_group(self):
        return True

    def create_parser(self, type: Callable[..., ArgumentParser]) -> ArgumentParser:
        parser = type(self.name, help=self.description)
        parser.set_defaults(**{f"__subcommand_help_{id(self):x}__": parser.print_help})
        return parser

    def run(self, args: Namespace):
        attr_name = f"__subcommand_help_{id(self):x}__"
        assert hasattr(args, attr_name)
        func = getattr(args, attr_name)
        return func()


class _SubCommandMethod(SubCommand):

    def __init__(self, info: _SubCommandMethodInfo, target: Any, id: str = None, parent_id: str = None):
        super().__init__(
            id=id,
            parent_id=parent_id,
            name=info.name,
            description=info.kwargs.get("description", None) or info.kwargs.get("help", None) or ""
        )
        self.info = info
        self.target = target

    def create_parser(self, type: Callable[..., ArgumentParser]) -> ArgumentParser:

        actions = []
        method = getattr(self.target, self.info.func.__name__)
        parser = type(self.name, **self.info.kwargs)
        parser.set_defaults(**{f"__subcommand_actions_{id(self):x}__": actions})

        for argument in reversed(self.info.arguments):
            argument_args = argument.args
            argument_kwargs = dict(argument.kwargs)

            # 解析dest，把注解的参数和方法参数对应上
            dest = argument_kwargs.get("dest", None)
            if not dest:
                prefix_chars = parser.prefix_chars
                if not argument_args or len(argument_args) == 1 and argument_args[0][0] not in prefix_chars:
                    dest = argument_args[0]
                    argument_kwargs["required"] = __missing__  # 这种方式不能指定required，所以这里设置为MISSING
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

            # 验证一下dest是否在参数列表中，不在就报错
            signature = inspect.signature(method)
            if dest not in signature.parameters:
                raise SubCommandError(
                    f"Check subcommand argument error, "
                    f"{self.info} has no `{dest}` argument")

            # 根据方法参数的注解，设置一些默认值
            parameter = signature.parameters[dest]
            if "default" not in argument_kwargs:
                if parameter.default != signature.empty:
                    argument_kwargs.setdefault("default", parameter.default)
                    argument_kwargs.setdefault("required", False)
                else:
                    argument_kwargs.setdefault("required", True)
            if "action" not in argument_kwargs:
                if parameter.annotation != signature.empty:
                    if parameter.annotation in (int, float, str):
                        argument_kwargs.setdefault("type", parameter.annotation)
                    elif parameter.annotation == bool:
                        if argument_kwargs.get("default", False):
                            argument_kwargs.setdefault("action", "store_false")
                        else:
                            argument_kwargs.setdefault("action", "store_true")

            actions.append(parser.add_argument(
                *argument_args,
                **_filter_kwargs(argument_kwargs)
            ))

        return parser

    def run(self, args: Namespace):
        method = getattr(self.target, self.info.func.__name__)

        attr_name = f"__subcommand_actions_{id(self):x}__"
        assert hasattr(args, attr_name)
        actions = getattr(args, attr_name)

        method_args = []
        if self.info.pass_args:
            method_args.append(args)

        method_kwargs = dict()
        for action in actions:
            method_kwargs[action.dest] = getattr(args, action.dest)

        return method(*method_args, **method_kwargs)


class SubCommandWrapper(SubCommand):

    def __init__(self, command: "BaseCommand",
                 id: str = None, parent_id: str = None,
                 name: str = None, description: str = None):
        super().__init__(
            id=id or _join_id(command.parent, command.name),
            parent_id=parent_id or _join_id(command.parent),
            name=name or command.name,
            description=description or command.description
        )
        self.command = command

    def create_parser(self, type: Callable[..., ArgumentParser]) -> ArgumentParser:
        return self.command.create_parser(self.name, help=self.description, type=type)

    def run(self, args: Namespace):
        return self.command(args)


class SubCommandMixin:

    def walk_subcommands(self: "BaseCommand", target: Any, parent_id: str = None) -> Generator[SubCommand, None, None]:
        """
        根据target对象，遍历所有的子命令，规则如下：
        1. 如果target是SubCommand类型，则直接返回
        2. 如果target是list、tuple、set、generator类型，则递归遍历
        3. 如果target是模块类型，则遍历模块下的所有子命令
        4. 如果target是其他类型，则遍历target下的所有包含@subcommand注解的方法
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
                    parent_id=_join_id(parent_id, target.parent_id)
                )
            else:
                yield SubCommandGroup(
                    target.command_name, target.command_description,
                    id=_join_id(parent_id, target.id),
                    parent_id=_join_id(parent_id, target.parent_id)
                )

        elif isinstance(target, ModuleType):
            for c in iter_module_commands(target, onerror="warn"):
                if c.command:
                    yield SubCommandWrapper(
                        c.command,
                        id=_join_id(parent_id, c.id),
                        parent_id=_join_id(parent_id, c.parent_id)
                    )
                else:
                    yield SubCommandGroup(
                        c.command_name, c.command_description,
                        id=_join_id(parent_id, c.id),
                        parent_id=_join_id(parent_id, c.parent_id)
                    )

        else:
            subcommand_map: Dict[str, List[_SubCommandMethod]] = {}
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
                    info: _SubCommandMethodInfo = func.__subcommand_info__
                    subcommand = _SubCommandMethod(info, target, parent_id=parent_id)
                    subcommand_map.setdefault(subcommand.name, list())
                    subcommand_map[info.name].append(subcommand)

            command_infos: List[Tuple[int, _SubCommandMethod]] = []
            for name, subcommands in subcommand_map.items():
                command_infos.append((min([c.info.index for c in subcommands]), subcommands[0]))
            for _, subcommand in sorted(command_infos, key=lambda o: o[0]):
                yield subcommand

    def add_subcommands(
            self: "BaseCommand",
            parser: ArgumentParser = None,
            target: Any = None,
            required: bool = False,
            sort: bool = False,
    ) -> List[_SubCommandInfo]:
        """
        向parser中添加子命令，规则参考walk_subcommands方法
        """
        target = target or self
        target_parser = parser or self._argument_parser

        parsers = {}
        root_parser = parser.add_subparsers(metavar="COMMAND", help="Command Help")
        root_parser.required = required

        subcommand_infos: List[_SubCommandInfo] = []
        for subcommand in self.walk_subcommands(target):
            subcommand_info = _SubCommandInfo(subcommand)
            subcommand_infos.append(subcommand_info)

            parent_parser = root_parser
            if subcommand.has_parent:
                parent_parser = parsers.get(subcommand.parent_id, None)
                if not parent_parser:
                    raise SubCommandError(f"{subcommand} has no parent subparser")

            parser = subcommand.create_parser(type=parent_parser.add_parser)
            parser.set_defaults(**{f"__subcommand_{id(self):x}__": subcommand})
            self.init_global_arguments(parser)

            if subcommand.is_group:
                subparser = parser.add_subparsers(metavar="COMMAND", help="Command Help")
                subparser.required = False
                parsers[subcommand.id] = subparser

            # BaseCommand 类型单独处理，因为有可能在init_arguments中添加了子命令
            if isinstance(subcommand, SubCommandWrapper):
                sub_subcommand_infos = parser.get_default(f"__subcommands_{id(subcommand.command):x}__")
                if sub_subcommand_infos:
                    subcommand_info.children.extend(
                        sub_subcommand_infos
                    )

        if sort:
            map = {info.node.id: info for info in subcommand_infos}
            groups = {}
            for info in map.values():
                group = []
                parent_id = info.node.parent_id
                while parent_id in map:
                    parent = map[info.node.parent_id]
                    group.append(parent.node.id)
                    parent_id = parent.node.parent_id
                groups[info.node.id] = tuple(reversed(group))
            subcommand_infos = sorted(
                subcommand_infos,
                key=lambda x: (groups.get(x.node.id), x.node.name)
            )

        target_parser.set_defaults(**{f"__subcommands_{id(self):x}__": subcommand_infos})

        return subcommand_infos

    def parse_subcommand(self: "BaseCommand", args: Namespace) -> Optional[SubCommand]:
        """
        解析子出args中的子命令
        """
        name = f"__subcommand_{id(self):x}__"
        if hasattr(args, name):
            subcommand = getattr(args, name)
            if isinstance(subcommand, SubCommand):
                return subcommand

        return None

    def run_subcommand(self: "BaseCommand", args: Namespace) -> Optional[int]:
        """
        解析并运行args中的子命令
        """
        subcommand = self.parse_subcommand(args)
        if subcommand:
            return subcommand.run(args)

        raise SubCommandError("Not found subcommand")

    def print_subcommands(
            self: "BaseCommand",
            args: Namespace,
            root: SubCommand = None,
            max_level: int = None
    ) -> None:
        """
        打印args中的子命令
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

    def _make_subcommand_tree(
            self: "BaseCommand",
            tree: Tree,
            infos: List[_SubCommandInfo],
            root_id: str,
            max_level: Optional[int],
            sort: bool = False
    ) -> Tree:
        nodes: Dict[str, Tuple[Tree, int]] = {}
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

            if info.node.is_group or info.children:
                logo = "📖" if current_node_expanded else "📘"
                text = f"{logo} [underline red]{info.node.name}[/underline red]" \
                    if not self.environ.debug \
                    else f"{logo} [underline red]{info.node.name}[/underline red] [dim](group={info.node.is_group}, id={info.node.id})[/dim]"
                if info.node.description:
                    text = f"{text}: {info.node.description}"
                current_node = parent_node.add(text, expanded=current_node_expanded)
                nodes[info.node.id] = current_node, current_node_level
            else:
                text = f"👉 [bold red]{info.node.name}[/bold red]" \
                    if not self.environ.debug \
                    else f"👉 [bold red]{info.node.name}[/bold red] [dim](group={info.node.is_group}, id={info.node.id})[/dim]"
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
                    current_max_level,
                    info.node.is_group
                )

        return tree


class BaseCommand(SubCommandMixin, metaclass=abc.ABCMeta):

    @property
    def module(self) -> str:
        return self.__module__

    @property
    def name(self) -> str:
        """
        命令名
        """
        name = self.__module__
        index = name.rfind(".")
        if index >= 0:
            name = name[index + 1:]
        return name

    @property
    def parent(self) -> Optional[str]:
        """
        父命令名
        """
        return None

    @property
    def environ(self) -> "BaseEnviron":
        """
        环境信息
        """
        return environ

    @property
    def logger(self) -> logging.Logger:
        """
        日志记录器
        """
        return self.environ.logger

    @cached_property
    def description(self) -> str:
        """
        命令描述，默认从docstring中获取
        """
        return textwrap.dedent((self.__doc__ or "")).strip()

    @property
    def known_errors(self) -> List[Type[BaseException]]:
        """
        已知错误类型
        """
        return []

    @abc.abstractmethod
    def init_arguments(self, parser: ArgumentParser) -> None:
        """
        初始化参数，在调用create_parser时执行
        """
        pass

    @abc.abstractmethod
    def run(self, args: Namespace) -> Optional[int]:
        """
        业务逻辑入口
        """
        pass

    def create_parser(
            self,
            *args: Any,
            type: Callable[..., ArgumentParser] = ArgumentParser,
            formatter_class: Type[HelpFormatter] = RawDescriptionHelpFormatter,
            conflict_handler="resolve",
            **kwargs: Any
    ) -> ArgumentParser:
        """
        创建命令行解析器
        """
        description = kwargs.pop("description", None)
        if not description:
            description = self.description.strip()
            if description and self.environ.description != NotImplemented:
                description += os.linesep + os.linesep
                description += self.environ.description
        parser = type(
            *args,
            description=description,
            formatter_class=formatter_class,
            conflict_handler=conflict_handler,
            **kwargs
        )
        self.init_base_arguments(parser)
        self.init_arguments(parser)
        return parser

    @cached_property
    def _argument_parser(self) -> ArgumentParser:
        parser = self.create_parser()
        self.init_global_arguments(parser)
        return parser

    def init_base_arguments(self, parser: ArgumentParser) -> None:
        """
        初始化基础参数，在调用create_parser时执行
        """
        pass

    def init_global_arguments(self, parser: ArgumentParser) -> None:
        """
        初始化公共参数，会在命令本身和所有子命令中调用
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
                environ.debug = True
                environ.logger.setLevel(logging.DEBUG)

        class LogTimeAction(BooleanOptionalAction):

            def __call__(self, parser, namespace, values, option_string=None):
                if option_string in self.option_strings:
                    value = not option_string.startswith("--no-")
                    handler = LogHandler.get_instance()
                    if handler:
                        handler.show_time = value

        class LogLevelAction(BooleanOptionalAction):

            def __call__(self, parser, namespace, values, option_string=None):
                if option_string in self.option_strings:
                    value = not option_string.startswith("--no-")
                    handler = LogHandler.get_instance()
                    if handler:
                        handler.show_level = value

        group = parser.add_argument_group(title="log options")
        group.add_argument(f"{prefix}{prefix}verbose", action=VerboseAction, nargs=0, const=True, dest=SUPPRESS,
                           help="increase log verbosity")
        group.add_argument(f"{prefix}{prefix}silent", action=SilentAction, nargs=0, const=True, dest=SUPPRESS,
                           help="disable all log output")
        group.add_argument(f"{prefix}{prefix}debug", action=DebugAction, nargs=0, const=True, dest=SUPPRESS,
                           help=f"increase {self.environ.name}'s log verbosity, and enable debug mode")

        if LogHandler.get_instance():
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
        """
        main命令入口
        """
        return CommandMain(self, show_log_level=True, show_log_time=False)

    def __call__(self, args: Union[List[str], Namespace] = None) -> int:
        """
        内部调用命令入口
        """
        try:
            if not isinstance(args, Namespace):
                parser = self._argument_parser
                if auto_complete:
                    auto_complete.autocomplete(parser)
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

    def init_subcommands(self) -> Any:
        return self

    def init_arguments(self, parser: ArgumentParser) -> None:
        self.add_subcommands(
            parser=parser,
            target=self.init_subcommands(),
        )

    def run(self, args: Namespace) -> Optional[int]:
        subcommand = self.parse_subcommand(args)
        if not subcommand or subcommand.is_group:
            return self.print_subcommands(args, subcommand, max_level=2)
        return subcommand.run(args)


class CommandMain:

    def __init__(
            self,
            command: BaseCommand, *,
            show_log_time: bool = False,
            show_log_level: bool = False,
            exit_on_return: bool = True
    ):
        self._command = command
        self.show_log_level = show_log_level
        self.show_log_time = show_log_time
        self.exit_on_return = exit_on_return

    @property
    def command(self) -> BaseCommand:
        return self._command

    def init_logging(self):
        """
        初始化log
        """
        init_logging(
            level=logging.INFO,
            show_time=self.show_log_time,
            show_level=self.show_log_level,
        )

    def __call__(self, *args, **kwargs):
        """
        main命令入口
        """
        self.init_logging()

        try:
            result = self.command(*args, **kwargs)
        except SystemExit as e:
            result = e.code
        except (KeyboardInterrupt, EOFError) as e:
            error_type, error_message = e.__class__.__name__, str(e).strip()
            self.command.logger.error(
                f"{error_type}: {error_message}" if error_message else error_type,
                exc_info=True if self.command.environ.debug else None,
            )
            result = 1
        except:
            get_console().print_exception(show_locals=True) \
                if self.command.environ.debug \
                else self.command.logger.error(traceback.format_exc())
            result = 1

        if self.exit_on_return:
            sys.exit(result)

        return result
