#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Terminal UI: prompt/confirm/choose, progress bars, and logging delegates.

Logging handler ownership lives in ``core/_logging.py``; this module provides
UI helpers and thin compatibility wrappers (``init_logging``,
``get_log_handler``, ``is_rich_available``) that delegate there.
"""
import getpass
import os
import re
import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeVar

from linktools.types import MISSING
from linktools.errors import CliError

#  When True, prompt/confirm/choose never block for input.
_no_input = False


def set_no_input(enabled: bool = True) -> None:
    """Enable or disable non-interactive mode."""
    global _no_input
    _no_input = bool(enabled)


def is_no_input() -> bool:
    """Return whether non-interactive mode is active."""
    return _no_input

if TYPE_CHECKING:
    from typing import TextIO
    from collections.abc import Iterable
    from rich.console import ConsoleRenderable, Console
    from rich.prompt import PromptBase
    from rich.text import Text, TextType
    from rich.progress import Progress, Task

    T = TypeVar("T")

    PromptType = TypeVar("PromptType", bound=PromptBase)
    PromptResultType = str | int | float | bool

_rich_available: "bool | None" = None


def is_rich_available() -> bool:
    """Return whether rich is importable and not suppressed (argcomplete)."""
    global _rich_available
    if _rich_available is None:
        from linktools.cli.argparse import ArgParseComplete
        if ArgParseComplete.is_invocation():
            _rich_available = False
        else:
            try:
                import rich  # noqa
                _rich_available = True
            except ImportError:
                _rich_available = False
    return _rich_available


def init_logging(
    level: int = 20,
    show_level: bool = False,
    show_time: bool = False,
    log_file: "str | None" = None,
) -> None:
    """Initialize root logging through the LoggingManager.

    Delegates to ``environ.logging.configure``; the 20 default matches
    ``logging.INFO`` without importing ``logging`` at module scope.
    """
    from linktools.core import environ
    environ.logging.configure(
        level=level,
        log_file=log_file,
        rich=is_rich_available(),
        show_level=show_level,
        show_time=show_time,
    )


def get_log_handler() -> "Any | None":
    """Return the active linktools log handler, if one is installed."""
    from linktools.core import environ
    return environ.logging.get_handler()


class _FakeProgress:
    """Text-based progress bar for when rich is not installed."""

    _BAR_WIDTH = 20

    def __init__(self):
        self._tasks: "dict[int, dict[str, Any]]" = {}
        self._next_id = 0
        self._is_tty = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
        self._last_line_len = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._finish()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self._finish()

    def _finish(self):
        if self._is_tty and self._last_line_len > 0:
            sys.stderr.write("\n")
            sys.stderr.flush()
            self._last_line_len = 0

    def add_task(self, description: str = "", total: "float | None" = None, **kwargs: "Any") -> int:
        task_id = self._next_id
        self._next_id += 1
        self._tasks[task_id] = {
            "description": description,
            "total": total,
            "completed": 0,
            "fields": {k: v for k, v in kwargs.items()},
        }
        self._render(task_id)
        return task_id

    def update(self, task_id: int, **kwargs: "Any") -> None:
        task = self._tasks.get(task_id)
        if task is None:
            return
        if "description" in kwargs:
            task["description"] = kwargs.pop("description")
        if "total" in kwargs:
            task["total"] = kwargs.pop("total")
        if "completed" in kwargs:
            task["completed"] = kwargs.pop("completed") or 0
        if "advance" in kwargs:
            task["completed"] = (task["completed"] or 0) + kwargs.pop("advance")
        # remaining kwargs are custom fields
        task["fields"].update(kwargs)
        self._render(task_id)

    def advance(self, task_id: int, advance: float = 1) -> None:
        task = self._tasks.get(task_id)
        if task is None:
            return
        task["completed"] = (task["completed"] or 0) + advance
        self._render(task_id)

    def _render(self, task_id: int):
        task = self._tasks.get(task_id)
        if task is None:
            return

        description = task["description"] or ""
        completed = task["completed"] or 0
        total = task["total"]

        parts = []
        if description:
            parts.append(description)

        if total:
            filled = int(self._BAR_WIDTH * completed / total)
            bar = "=" * filled + "-" * (self._BAR_WIDTH - filled)
            pct = f"{100 * completed / total:.1f}%"
            parts.append(f"[{bar}]")
            parts.append(pct)
            parts.append(f"({self._fmt_size(completed)}/{self._fmt_size(total)})")
        else:
            parts.append(f"({self._fmt_size(completed)})")

        for v in task["fields"].values():
            if v:
                # strip rich markup tags
                parts.append(re.sub(r"\[[^\]]*\]", "", str(v)).strip())

        line = " ".join(p for p in parts if p)

        if self._is_tty:
            # overwrite current line
            clear = " " * max(0, self._last_line_len - len(line))
            sys.stderr.write(f"\r{line}{clear}")
            sys.stderr.flush()
            self._last_line_len = len(line)
        else:
            sys.stderr.write(f"{line}\n")
            sys.stderr.flush()

    @staticmethod
    def _fmt_size(n: float) -> str:
        n = float(n)
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024:
                return f"{n:.1f}{unit}"
            n /= 1024
        return f"{n:.1f}TB"


def _get_log_column():
    from rich.table import Column
    from rich.text import Text
    from rich.progress import ProgressColumn

    class _LogColumn(ProgressColumn):

        def __init__(self):
            super().__init__(table_column=Column(no_wrap=True))

        def render(self, task: "Task" = None) -> "str | Text":
            result = Text()

            handler = get_log_handler()
            if handler and handler.show_time:
                if len(result) > 0:
                    result.append(" ")
                result.append(handler.make_time_text())

            if handler and handler.show_level:
                if len(result) > 0:
                    result.append(" ")
                result.append(handler.make_level_text(20))

            return result

    return _LogColumn()


def create_progress(*fields: str, transfer: bool = False) -> "_FakeProgress | Progress":
    """Create a progress renderer with optional task fields.

    Args:
        fields (str): Extra per-task fields to render as columns.
        transfer (bool): If True, render transfer-oriented columns
            (size, speed, ETA) in addition to the description and bar.
            Suitable for both uploads and downloads.

    Returns:
        Any: The operation result.
    """
    if not is_rich_available():
        return _FakeProgress()

    from rich.progress import Progress, TextColumn, BarColumn

    columns = []

    handler = get_log_handler()
    if handler and (handler.show_time or handler.show_level):
        columns.append(_get_log_column())

    columns.extend([
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
    ])

    if transfer:
        from rich.progress import DownloadColumn, TransferSpeedColumn, TaskProgressColumn, TimeRemainingColumn

        columns.extend([
            DownloadColumn(),
            TransferSpeedColumn(),
            TaskProgressColumn(),
            TextColumn("eta"),
            TimeRemainingColumn(),
        ])

    for field in fields:
        columns.append(TextColumn(f"{{task.fields[{field}]}}"))

    return Progress(*columns)


def _create_prompt_class(type: "type[PromptResultType]", allow_empty: bool) -> "type[PromptType]":
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
                stream: "TextIO | None" = None,
        ) -> str:

            prefix = []
            prefix_len = 0

            handler = get_log_handler()
            if handler and handler.show_time:
                time = handler.make_time_text()
                prefix.append(time)
                prefix_len += time.cell_len + 1
            if handler and handler.show_level:
                level = handler.make_level_text(30, ">")
                prefix.append(level)
                prefix_len += level.cell_len + 1

            lines = prompt.split(include_separator=True, allow_blank=True)
            console.print(*(*prefix, lines[0]), sep=" ", end="")
            for i in range(1, len(lines)):
                lines[i].pad_left(prefix_len)
                console.print(lines[i], new_line_start=True, end="")

            return console.input(password=password, stream=stream)

        def on_validate_error(self, value: str, error: "InvalidResponse") -> None:
            prefix = Text("")
            handler = get_log_handler()
            if handler and handler.show_time:
                prefix = prefix + handler.make_time_text() + " "
            if handler and handler.show_level:
                prefix = prefix + handler.make_level_text(40, ">") + " "
            self.console.print(prefix, error, sep="")

        def process_response(self, value: str) -> "PromptType":
            value = value.strip()
            if not allow_empty and not value:
                raise InvalidResponse(self.validate_error_message)
            return super().process_response(value)

    return RichPrompt


def _plain_prompt(
        prompt_text: str,
        type: "type" = str,
        default=MISSING,
        allow_empty: bool = False,
        choices: "list[str] | None" = None,
        password: bool = False,
        show_default: bool = True,
        show_choices: bool = True,
):
    suffix_parts = []
    if choices and show_choices:
        suffix_parts.append(f"[{'/'.join(choices)}]")
    if default is not MISSING and show_default:
        suffix_parts.append(f"(default: {default})")
    full_prompt = prompt_text
    if suffix_parts:
        full_prompt += " " + " ".join(suffix_parts)
    full_prompt += ": "

    while True:
        try:
            value = getpass.getpass(full_prompt) if password else input(full_prompt)
        except (EOFError, KeyboardInterrupt):
            raise

        value = value.strip()

        if not value:
            if default is not MISSING:
                return default
            if allow_empty:
                return type()
            print("Please enter a value.")
            continue

        if choices and value not in choices:
            print(f"Invalid choice. Choose from: {', '.join(choices)}")
            continue

        try:
            return type(value)
        except (ValueError, TypeError):
            print("Invalid value.")
            continue


def _plain_confirm(
        prompt_text: str,
        default=MISSING,
        show_default: bool = True,
) -> bool:
    while True:
        if default is not MISSING and show_default:
            hint = " [Y/n]" if default else " [y/N]"
        else:
            hint = " [y/n]"
        try:
            value = input(prompt_text + hint + ": ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            raise
        if not value:
            if default is not MISSING:
                return bool(default)
            continue
        if value in ('y', 'yes'):
            return True
        if value in ('n', 'no'):
            return False
        print("Please enter 'y' or 'n'.")


def _plain_choose(
        prompt_text: str,
        choices: "Iterable | Dict",
        title: str = None,
        default=MISSING,
        show_default: bool = True,
        show_choices: bool = True,
):
    if isinstance(choices, dict):
        keys = tuple(choices.keys())
        texts = [str(choices[key]) for key in keys]
    else:
        keys = tuple(choices)
        texts = [str(c) for c in keys]

    begin_id = 1
    tip_id = 0
    default_id = None
    if default is not MISSING and default in keys:
        tip_id = default_id = keys.index(default)

    if title:
        print(title)
    for i, text in enumerate(texts):
        prefix = ">> " if i == tip_id else "   "
        print(f"{prefix}{i + begin_id}: {text}")

    range_str = f"[{begin_id}~{len(texts) + begin_id - 1}]" if len(texts) > 1 else f"[{begin_id}]"
    full_prompt = prompt_text
    if show_choices:
        full_prompt += f" {range_str}"
    if default_id is not None and show_default:
        full_prompt += f" (default: {default_id + begin_id})"
    full_prompt += ": "

    valid = [str(i) for i in range(begin_id, len(texts) + begin_id)]
    while True:
        try:
            value = input(full_prompt).strip()
        except (EOFError, KeyboardInterrupt):
            raise
        if not value and default_id is not None:
            return keys[default_id]
        if value in valid:
            return keys[int(value) - begin_id]
        print(f"Please enter a number between {begin_id} and {len(texts) + begin_id - 1}.")


def prompt(
        prompt: str,
        type: "type[PromptResultType]" = str,
        default: "PromptResultType" = MISSING,
        allow_empty: bool = False,
        choices: "list[str] | None" = None,
        password: bool = False,
        show_default: bool = True,
        show_choices: bool = True
) -> "PromptResultType":
    """Prompt for a typed value using rich when available.

    Args:
        prompt (str): The prompt value.
        type (type[PromptResultType]): Target type used to cast the value.
        default (PromptResultType): Value returned when no explicit value is available.
        allow_empty (bool): The allow_empty value.
        choices (Optional[List[str]]): The choices value.
        password (bool): Password used for authentication.
        show_default (bool): The show_default value.
        show_choices (bool): The show_choices value.

    Returns:
        PromptResultType: The operation result.
    """
    if _no_input:
        if default is not MISSING:
            return default
        raise CliError("prompt requires interaction but no-input mode is active: " + prompt)
    if not is_rich_available():
        return _plain_prompt(
            prompt, type=type, default=default, allow_empty=allow_empty,
            choices=choices, password=password, show_default=show_default,
            show_choices=show_choices,
        )
    return _create_prompt_class(type, allow_empty=allow_empty).ask(
        prompt,
        password=password,
        choices=choices,
        default=str(default) if default is not MISSING else ...,
        show_default=show_default,
        show_choices=show_choices
    )


def choose(
        prompt: str,
        choices: "Iterable[T] | dict[T, Any]",
        title: "str | None" = None,
        default: "T" = MISSING,
        show_default: bool = True,
        show_choices: bool = True
) -> "T":
    """Prompt the user to choose one item from a list or mapping.

    Args:
        prompt (str): The prompt value.
        choices (Union[Iterable[T], Dict[T, Any]]): The choices value.
        title (str): The title value.
        default (T): Value returned when no explicit value is available.
        show_default (bool): The show_default value.
        show_choices (bool): The show_choices value.

    Returns:
        T: The operation result.
    """
    if _no_input:
        if default is not MISSING:
            return default
        raise CliError("choose requires interaction but no-input mode is active: " + prompt)
    if not is_rich_available():
        return _plain_choose(
            prompt, choices, title=title, default=default,
            show_default=show_default, show_choices=show_choices,
        )

    from rich.text import Text

    if isinstance(choices, dict):
        keys = tuple(choices.keys())
        texts = [str(choices[key]) for key in keys]
    else:
        keys = tuple(choices)
        texts = [str(choice) for choice in keys]

    tip_id = 0
    default_id = None
    if default is not MISSING and default in keys:
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
        default: "PromptResultType" = MISSING,
        show_default: bool = True,
) -> bool:
    """Prompt the user for a yes-or-no confirmation.

    Args:
        prompt (str): The prompt value.
        default (PromptResultType): Value returned when no explicit value is available.
        show_default (bool): The show_default value.

    Returns:
        bool: The operation result.
    """
    if _no_input:
        return default if default is not MISSING else True
    if not is_rich_available():
        return _plain_confirm(prompt, default=default, show_default=show_default)
    return _create_prompt_class(bool, allow_empty=False).ask(
        prompt,
        default=str(default) if default is not MISSING else ...,
        show_default=show_default,
    )
