#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : logging.py
@time    : 2020/03/22
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
import logging
import os
from abc import ABCMeta, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Union, List, Dict, Type, TypeVar, TextIO, Iterable, Any

from .metadata import __missing__

if TYPE_CHECKING:
    from rich.console import ConsoleRenderable, Console
    from rich.prompt import PromptBase
    from rich.text import Text, TextType

    T = TypeVar("T")

    PromptType = TypeVar("PromptType", bound=PromptBase)
    PromptResultType = TypeVar("PromptResultType", str, int, float, bool)


class _LogHandlerMixin(metaclass=ABCMeta):

    @property
    @abstractmethod
    def show_level(self) -> bool:
        ...

    @show_level.setter
    @abstractmethod
    def show_level(self, value: bool):
        ...

    @property
    @abstractmethod
    def show_time(self) -> bool:
        ...

    @show_time.setter
    @abstractmethod
    def show_time(self, value: bool):
        ...

    @abstractmethod
    def make_time_text(self, time: "float | datetime | None" = None, format: str = None, style: str = None) -> "Text":
        ...

    @abstractmethod
    def make_level_text(self, level_no: int, level_name: str = None, style: str = None) -> "Text":
        ...


def _get_rich_log_handler_class():
    from rich.logging import RichHandler
    from rich.text import Text

    class LogHandler(RichHandler, _LogHandlerMixin):

        def __init__(self, show_level: bool, show_time: bool):
            super().__init__(
                show_path=False,
                show_level=show_level,
                show_time=show_time,
                omit_repeated_times=False,
                log_time_format=self.make_time_text
                # markup=True,
                # highlighter=NullHighlighter()
            )

            self._styles = {
                logging.DEBUG: {
                    "level": "black on blue",
                    "message": "deep_sky_blue1",
                },
                logging.INFO: {
                    "level": "black on green",
                    "message": None,
                },
                logging.WARNING: {
                    "level": "black on yellow",
                    "message": "magenta1",
                },
                logging.ERROR: {
                    "level": "black on red1",
                    "message": "red1",
                },
                logging.CRITICAL: {
                    "level": "black on red1",
                    "message": "red1",
                },
            }

        @property
        def show_level(self):
            return self._log_render.show_level

        @show_level.setter
        def show_level(self, value: bool):
            self._log_render.show_level = value

        @property
        def show_time(self):
            return self._log_render.show_time

        @show_time.setter
        def show_time(self, value: bool):
            self._log_render.show_time = value

        def make_time_text(self, time: "float | datetime | None" = None, format: str = None, style: str = None) -> "Text":
            if not time:
                time = datetime.now()
            elif isinstance(time, (int, float)):
                time = datetime.fromtimestamp(time)
            if not style:
                style = "log.time"
            if not format:
                if self.formatter:
                    format = self.formatter.datefmt
                if not format:
                    format = "[%x %X]"
            return Text(time.strftime(format), style=style)

        def make_level_text(self, level_no: int, level_name: str = None, style: str = None) -> "Text":
            if not level_name:
                level_name = logging.getLevelName(level_no)
            if not style:
                style = self.get_level_style(level_no)
                if not style:
                    style = "log.level"
            return Text(f" {level_name[:1].upper()} ", style=style)

        def get_time_style(self, level_no):
            style = self._styles.get(level_no)
            if style:
                return style.get("time")
            return None

        def get_level_style(self, level_no):
            style = self._styles.get(level_no)
            if style:
                return style.get("level")
            return None

        def get_message_style(self, level_no):
            style = self._styles.get(level_no)
            if style:
                return style.get("message")
            return None

        def get_level_text(self, record: logging.LogRecord) -> "Text":
            level_name = record.levelname
            level_no = record.levelno
            return self.make_level_text(level_no, level_name)

        def render_message(self, record: logging.LogRecord, message: str) -> "ConsoleRenderable":
            indent = getattr(record, "indent", 0)
            if indent > 0:
                message = " " * indent + message
                message = message.replace(os.linesep, os.linesep + " " * indent)

            use_markup = getattr(record, "markup", self.markup)
            style = getattr(record, "style", self.get_message_style(record.levelno))
            message_text = Text.from_markup(message, style=style) if use_markup else Text(message, style=style)

            highlighter = getattr(record, "highlighter", False)
            if highlighter and self.highlighter:
                message_text = self.highlighter(message_text)

            return message_text

    return LogHandler


def _get_fake_log_handler_class():
    class LogHandler(logging.Handler, _LogHandlerMixin):

        def __init__(self, show_level: bool, show_time: bool):
            super().__init__()
            self._show_level = show_level
            self._show_time = show_time

        @property
        def show_level(self):
            return self._show_level

        @show_level.setter
        def show_level(self, value: bool):
            self._show_level = value

        @property
        def show_time(self):
            return self._show_time

        @show_time.setter
        def show_time(self, value: bool):
            self._show_time = value

        def make_time_text(self, time: "float | datetime | None" = None, format: str = None, style: str = None) -> "Text":
            from rich.text import Text
            return Text("")

        def make_level_text(self, level_no: int, level_name: str = None, style: str = None) -> "Text":
            from rich.text import Text
            return Text("")

    return LogHandler


def init_logging(level: int = logging.INFO, show_level: bool = False, show_time: bool = False, force: bool = False):
    from .cli.argparse import ArgParseComplete

    if ArgParseComplete.is_invocation():
        log_handler_class = _get_fake_log_handler_class()
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[log_handler_class(show_level=show_level, show_time=show_time)],
            force=force,
        )
        return

    from rich import get_console

    if get_console().is_terminal:
        log_handler_class = _get_rich_log_handler_class()
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[log_handler_class(show_level=show_level, show_time=show_time)],
            force=force,
        )

    else:
        items = []
        if show_time:
            items.append("[%(asctime)s]")
        if show_level:
            items.append("%(levelname)s")
        items.extend(["%(module)s", "%(funcName)s", "%(message)s"])
        logging.basicConfig(
            level=level,
            format=" ".join(items),
            datefmt="%H:%M:%S",
            force=force,
        )


def get_log_handler() -> "Optional[_LogHandlerMixin]":
    c = logging.getLogger()
    while c:
        if c.handlers:
            for handler in c.handlers:
                if isinstance(handler, _LogHandlerMixin):
                    return handler
        if not c.propagate:
            return None
        else:
            c = c.parent
    return None


def _get_log_column():
    from rich.table import Column
    from rich.text import Text
    from rich.progress import Task, ProgressColumn

    class _LogColumn(ProgressColumn):

        def __init__(self):
            super().__init__(table_column=Column(no_wrap=True))

        def render(self, task: Task = None) -> "Union[str, Text]":
            result = Text()

            handler = get_log_handler()
            if handler and handler.show_time:
                if len(result) > 0:
                    result.append(" ")
                result.append(handler.make_time_text())

            if handler and handler.show_level:
                if len(result) > 0:
                    result.append(" ")
                result.append(handler.make_level_text(logging.INFO))

            return result

    return _LogColumn()


def create_simple_progress(*fields: str):
    from rich.progress import Progress, TextColumn, BarColumn

    columns = []

    handler = get_log_handler()
    if handler and (handler.show_time or handler.show_level):
        columns.append(_get_log_column())

    columns.extend([
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
    ])

    for field in fields:
        columns.append(TextColumn(f"{{task.fields[{field}]}}"))

    return Progress(*columns)


def create_progress():
    from rich.progress import Progress, TextColumn, BarColumn, DownloadColumn, \
        TransferSpeedColumn, TaskProgressColumn, TimeRemainingColumn

    columns = []

    handler = get_log_handler()
    if handler and (handler.show_time or handler.show_level):
        columns.append(_get_log_column())

    columns.extend([
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TaskProgressColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
    ])

    return Progress(*columns)


def _create_prompt_class(type: "Type[PromptResultType]", allow_empty: bool) -> "Type[PromptType]":
    from rich.text import Text
    from rich.prompt import Prompt, IntPrompt, InvalidResponse, FloatPrompt, Confirm

    prompt_types = {str: Prompt, int: IntPrompt, float: FloatPrompt, bool: Confirm}
    prompt_type = prompt_types.get(type, None)
    if prompt_type is None:
        raise TypeError(f"Unknown prompt type: {prompt_type}")

    class RichPrompt(prompt_type):

        @classmethod
        def get_input(
                cls,
                console: "Console",
                prompt: "TextType",
                password: bool,
                stream: Optional[TextIO] = None,
        ) -> str:

            prefix = []
            prefix_len = 0

            handler = get_log_handler()
            if handler and handler.show_time:
                time = handler.make_time_text()
                prefix.append(time)
                prefix_len += time.cell_len + 1
            if handler and handler.show_level:
                level = handler.make_level_text(logging.WARNING, "↳")
                prefix.append(level)
                prefix_len += level.cell_len + 1

            lines = prompt.split(include_separator=True, allow_blank=True)
            console.print(*(*prefix, lines[0]), sep=" ", end="")
            for i in range(1, len(lines)):
                lines[i].pad_left(prefix_len)
                console.print(lines[i], new_line_start=True, end="")

            return console.input(password=password, stream=stream)

        def on_validate_error(self, value: str, error: InvalidResponse) -> None:
            prefix = Text("")
            handler = get_log_handler()
            if handler and handler.show_time:
                prefix = prefix + handler.make_time_text() + " "
            if handler and handler.show_level:
                prefix = prefix + handler.make_level_text(logging.ERROR, "↳") + " "
            self.console.print(prefix, error, sep="")

        def process_response(self, value: str) -> "PromptType":
            value = value.strip()
            if not allow_empty and not value:
                raise InvalidResponse(self.validate_error_message)
            return super().process_response(value)

    return RichPrompt


def prompt(
        prompt: str,
        type: "Type[PromptResultType]" = str,
        default: "PromptResultType" = __missing__,
        allow_empty: bool = False,
        choices: Optional[List[str]] = None,
        password: bool = False,
        show_default: bool = True,
        show_choices: bool = True
) -> "PromptResultType":
    return _create_prompt_class(type, allow_empty=allow_empty).ask(
        prompt,
        password=password,
        choices=choices,
        default=default if default is not __missing__ else ...,
        show_default=show_default,
        show_choices=show_choices
    )


def choose(
        prompt: str,
        choices: "Union[Iterable[T], Dict[T, Any]]",
        title: str = None,
        default: "T" = __missing__,
        show_default: bool = True,
        show_choices: bool = True
) -> "T":
    from rich.text import Text

    if isinstance(choices, dict):
        keys = tuple(choices.keys())
        texts = [str(choices[key]) for key in keys]
    else:
        keys = tuple(choices)
        texts = [str(choice) for choice in choices]

    tip_id = 0
    default_id = None
    if default is not __missing__ and default in keys:
        tip_id = default_id = keys.index(default)

    begin_id = 1
    text = Text()
    if title:
        text.append(f"{title}{os.linesep}")
    for i in range(len(texts)):
        text.append(f"{'>> ' if i == tip_id else '   '}")
        text.append(f"{f'{i + begin_id}:':2} ", "prompt.choices")
        text.append(f"{texts[i]}{os.linesep}")
    text.append(prompt)
    if show_choices:
        text.append(" ")
        text.append(f"[{begin_id}~{len(texts) + begin_id - 1}]" if len(texts) > 1 else f"[{begin_id}]",
                    "prompt.choices")

    index = _create_prompt_class(int, allow_empty=False).ask(
        text,
        choices=[str(i) for i in range(begin_id, len(texts) + begin_id, 1)],
        default=default_id + begin_id if default_id is not None else ...,
        show_default=show_default,
        show_choices=False,
    ) - begin_id

    return keys[index]


def confirm(
        prompt: str,
        default: "PromptResultType" = __missing__,
        show_default: bool = True,
) -> bool:
    return _create_prompt_class(bool, allow_empty=False).ask(
        prompt,
        default=default if default is not __missing__ else ...,
        show_default=show_default,
    )
