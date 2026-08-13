#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Unified logging management and handler ownership.

Business modules obtain their logger through ``environ.get_logger(name)``; the
:class:`LoggingManager` owns everything else -- the secret-redaction filter,
thread-local log context, the two-phase bootstrap/configure lifecycle,
third-party logger bridging (paramiko), and the process-global linktools log
handler.
"""
import contextlib
import datetime
import logging
import os
import re
import threading
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable, Iterator, Pattern

# ---------------------------------------------------------------------------
# Global LogRecordFactory manager
# ---------------------------------------------------------------------------

_factory_lock = threading.Lock()
_original_factory = logging.getLogRecordFactory()
_active_redactors: "dict[int, Callable[[logging.LogRecord], None]]" = {}
_factory_installed = False

# Process-global owned-handler state.
_handler_lock = threading.Lock()
_bootstrap_handler: "Any | None" = None
_active_handler: "Any | None" = None
_active_handler_mode: "str | None" = None  # None / plain / rich / file


def _chained_factory(*args, **kwargs):
    record = _original_factory(*args, **kwargs)
    for redactor in list(_active_redactors.values()):
        redactor(record)
    return record


def _register_redactor(key: int, redactor: "Callable[[logging.LogRecord], None]") -> None:
    global _factory_installed
    with _factory_lock:
        _active_redactors[key] = redactor
        if not _factory_installed:
            logging.setLogRecordFactory(_chained_factory)
            _factory_installed = True


def _unregister_redactor(key: int) -> None:
    global _factory_installed
    with _factory_lock:
        _active_redactors.pop(key, None)
        if not _active_redactors and _factory_installed:
            logging.setLogRecordFactory(_original_factory)
            _factory_installed = False


__all__ = ["LoggingManager"]

_BUILTIN_REDACTORS: "list[tuple[Pattern[str], str]]" = [
    (re.compile(r"(://)[^\s/]*@"), r"://***@"),
    (re.compile(
        r"(?i)([?&](?:access[_-]?token|api[_-]?key|apikey|token|key|signature|"
        r"secret|password|passwd|pwd)=)([^&\s#]+)"), r"\1***"),
    (re.compile(
        r"(?i)((?:authorization|set[_-]?cookie|cookie|"
        r"[a-z0-9 _.-]{0,24}(?:token|password|passwd|pwd|secret[_\s-]?key|secret|"
        r"api[_\s-]?key|apikey|access[_\s-]?key|signature|private[_\s-]?key))"
        r"\s*[:=]\s*)([^\r\n]*)"), r"\1***"),
    (re.compile(r"(?i)(sshpass\s+(?:-p\s+|--password[=\s]))(\S+)"), r"\1***"),
]


# ---------------------------------------------------------------------------
# Plain-text fallback (no rich dependency)
# ---------------------------------------------------------------------------

class _FakeText:
    """Minimal Text substitute for when rich is not installed."""

    def __init__(self, text: "Any" = "", style: "Any" = None):
        self._text = str(text)

    def __len__(self):
        return len(self._text)

    def __str__(self):
        return self._text

    def __add__(self, other):
        return _FakeText(self._text + str(other))

    @property
    def cell_len(self) -> int:
        return len(self._text)

    @classmethod
    def from_markup(cls, text: str, style: "str | None" = None) -> "_FakeText":
        clean = re.sub(r'\[/?[^\]]*\]', '', str(text))
        return cls(clean)

    def append(self, text: str, style: "str | None" = None) -> "_FakeText":
        self._text += str(text)
        return self

    def split(self, separator: "str | None" = None, include_separator: bool = False, allow_blank: bool = False) -> "list[_FakeText]":
        parts = self._text.split('\n')
        return [_FakeText(p) for p in parts]

    def pad_left(self, n: int) -> "_FakeText":
        self._text = " " * n + self._text
        return self


class _LogHandlerMixin(metaclass=ABCMeta):

    @property
    @abstractmethod
    def show_level(self) -> bool:
        ...

    @property
    @abstractmethod
    def show_time(self) -> bool:
        ...

    @abstractmethod
    def make_time_text(self, time: "float | datetime.datetime | None" = None, format: "str | None" = None, style: "str | None" = None) -> "Any":
        ...

    @abstractmethod
    def make_level_text(self, level_no: int, level_name: "str | None" = None, style: "str | None" = None) -> "Any":
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
                log_time_format=self.make_time_text,
            )
            self._show_level = show_level
            self._show_time = show_time

            self._styles = {
                logging.DEBUG: {"level": "black on blue", "message": "deep_sky_blue1"},
                logging.INFO: {"level": "black on green", "message": None},
                logging.WARNING: {"level": "black on yellow", "message": "magenta1"},
                logging.ERROR: {"level": "black on red1", "message": "red1"},
                logging.CRITICAL: {"level": "black on red1", "message": "red1"},
            }

        @property
        def show_level(self) -> bool:
            return self._show_level

        @property
        def show_time(self) -> bool:
            return self._show_time

        def make_time_text(self, time: "float | datetime.datetime | None" = None, format: "str | None" = None, style: "str | None" = None) -> "Text":
            if not time:
                time = datetime.datetime.now()
            elif isinstance(time, (int, float)):
                time = datetime.datetime.fromtimestamp(time)
            if not style:
                style = "log.time"
            if not format:
                if self.formatter:
                    format = self.formatter.datefmt
                if not format:
                    format = "[%X]"
            return Text(time.strftime(format), style=style)

        def make_level_text(self, level_no: int, level_name: "str | None" = None, style: "str | None" = None) -> "Text":
            if not level_name:
                level_name = logging.getLevelName(level_no)
            if not style:
                style = self.get_level_style(level_no)
                if not style:
                    style = "log.level"
            return Text(f" {level_name[:1].upper()} ", style=style)

        def get_time_style(self, level_no: int) -> "str | None":
            style = self._styles.get(level_no)
            return style.get("time") if style else None

        def get_level_style(self, level_no: int) -> "str | None":
            style = self._styles.get(level_no)
            return style.get("level") if style else None

        def get_message_style(self, level_no: int) -> "str | None":
            style = self._styles.get(level_no)
            return style.get("message") if style else None

        def get_level_text(self, record: "logging.LogRecord") -> "Text":
            return self.make_level_text(record.levelno, record.levelname)

        def render_message(self, record: "logging.LogRecord", message: str) -> "Any":
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


def _get_plain_log_handler_class():
    class LogHandler(logging.StreamHandler, _LogHandlerMixin):

        def __init__(self, show_level: bool, show_time: bool):
            super().__init__()
            self._show_level = show_level
            self._show_time = show_time
            items = []
            if show_time:
                items.append("[%(asctime)s]")
            if show_level:
                items.append("%(levelname)s")
            items.extend(["%(module)s", "%(funcName)s", "%(message)s"])
            self.setFormatter(logging.Formatter(" ".join(items), datefmt="%H:%M:%S"))

        @property
        def show_level(self) -> bool:
            return self._show_level

        @property
        def show_time(self) -> bool:
            return self._show_time

        def make_time_text(self, time: "float | datetime.datetime | None" = None, format: "str | None" = None, style: "str | None" = None) -> _FakeText:
            if not time:
                time = datetime.datetime.now()
            elif isinstance(time, (int, float)):
                time = datetime.datetime.fromtimestamp(time)
            if not format:
                format = "[%X]"
            return _FakeText(time.strftime(format))

        def make_level_text(self, level_no: int, level_name: "str | None" = None, style: "str | None" = None) -> _FakeText:
            if not level_name:
                level_name = logging.getLevelName(level_no)
            return _FakeText(f" {level_name[:1].upper()} ")

    return LogHandler


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class _LocalContext(threading.local):

    def __init__(self) -> None:
        super().__init__()
        self.fields: "dict[str, Any]" = {}


class LoggingManager(object):

    def __init__(self, environ: "Any | None" = None) -> None:
        self._environ = environ
        self._secrets: "list[str]" = []
        self._patterns: "list[tuple[Pattern[str], str]]" = list(_BUILTIN_REDACTORS)
        self._local = _LocalContext()
        self._installed = False
        self._bootstrapped = False

    # -- redaction ----------------------------------------------------------

    def register_secret(self, value: "Any") -> None:
        if isinstance(value, str) and value:
            self._secrets.append(value)

    def register_redactor(
        self, pattern: "str | Pattern[str]", repl: str = "***"
    ) -> None:
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
        self._patterns.append((pattern, repl))

    def redact(self, text: "Any") -> "Any":
        if not isinstance(text, str):
            return text
        for regex, repl in self._patterns:
            text = regex.sub(repl, text)
        for secret in self._secrets:
            if secret:
                text = text.replace(secret, "***")
        return text

    # -- context -----------------------------------------------------------

    def current_context(self) -> "dict[str, Any]":
        return dict(self._local.fields)

    @contextlib.contextmanager
    def context(self, **fields: "Any") -> "Iterator[None]":
        saved = dict(self._local.fields)
        self._local.fields.update(fields)
        try:
            yield
        finally:
            self._local.fields = saved

    # -- redaction installation (record factory) ---------------------------

    def install_filter(self) -> None:
        if self._installed:
            return
        manager = self

        def _redact(record: "logging.LogRecord") -> None:
            for key, value in manager.current_context().items():
                if not hasattr(record, key):
                    setattr(record, key, value)
            try:
                message = record.getMessage()
            except Exception:
                message = record.msg if isinstance(record.msg, str) else None
            if isinstance(message, str):
                record.msg = manager.redact(message)
                record.args = ()

        _register_redactor(id(self), _redact)
        self._installed = True

    def remove_filter(self) -> None:
        if self._installed:
            _unregister_redactor(id(self))
            self._installed = False

    # -- logger access -----------------------------------------------------

    def get_logger(self, name: "str | None" = None) -> "logging.Logger":
        self.install_filter()
        if name is None:
            name = getattr(self._environ, "name", None) or "linktools"
        return logging.getLogger(name)

    def set_level(self, name: str, level: int) -> None:
        logging.getLogger(name).setLevel(level)

    # -- handler ownership -------------------------------------------------

    def get_handler(self) -> "Any | None":
        handler = _active_handler
        if isinstance(handler, _LogHandlerMixin):
            return handler
        return None

    @property
    def show_time(self) -> bool:
        handler = self.get_handler()
        return handler.show_time if handler is not None else False

    @property
    def show_level(self) -> bool:
        handler = self.get_handler()
        return handler.show_level if handler is not None else False

    def _create_display_handler(self, show_level: bool, show_time: bool) -> "_LogHandlerMixin":
        if _active_handler_mode == "rich" or (_rich_terminal_available() and _active_handler_mode in (None, "rich")):
            return _get_rich_log_handler_class()(show_level=show_level, show_time=show_time)
        else:
            return _get_plain_log_handler_class()(show_level=show_level, show_time=show_time)

    def _replace_display_handler(self, show_level: bool, show_time: bool) -> None:
        """Rollback-safe display handler replacement.

        Constructs a complete candidate before mutating root/global state.
        On any failure the old handler, root configuration, and globals
        remain unchanged.
        """
        global _active_handler, _active_handler_mode

        old_mode = _active_handler_mode
        if old_mode not in ("plain", "rich"):
            return

        old_handler = _active_handler
        if not isinstance(old_handler, _LogHandlerMixin):
            return

        old_show_level = old_handler.show_level
        old_show_time = old_handler.show_time

        if show_level == old_show_level and show_time == old_show_time:
            return

        # Construct candidate first; if this fails, nothing changes.
        candidate = self._create_display_handler(show_level=show_level, show_time=show_time)

        root = logging.getLogger()
        with _handler_lock:
            try:
                root.addHandler(candidate)
            except Exception:
                try:
                    candidate.close()
                except Exception:
                    pass
                raise

            try:
                root.removeHandler(old_handler)
            except Exception:
                root.removeHandler(candidate)
                try:
                    candidate.close()
                except Exception:
                    pass
                raise

            _active_handler = candidate
            try:
                old_handler.close()
            except Exception:
                pass

    def set_show_time(self, value: bool) -> None:
        handler = self.get_handler()
        if handler is None:
            return
        self._replace_display_handler(
            show_level=handler.show_level,
            show_time=value,
        )

    def set_show_level(self, value: bool) -> None:
        handler = self.get_handler()
        if handler is None:
            return
        self._replace_display_handler(
            show_level=value,
            show_time=handler.show_time,
        )

    # -- two-phase lifecycle -----------------------------------------------

    def bootstrap(self) -> None:
        global _bootstrap_handler
        if self._bootstrapped:
            return
        root = logging.getLogger()
        with _handler_lock:
            if not root.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
                root.addHandler(handler)
                _bootstrap_handler = handler
        if root.level == logging.NOTSET:
            root.setLevel(logging.WARNING)
        self.install_filter()
        self._bootstrapped = True

    def configure(
        self,
        level: int = logging.INFO,
        log_file: "str | None" = None,
        rich: bool = True,
        show_level: bool = False,
        show_time: bool = False,
    ) -> None:
        """Phase 2 -- configure root logging with rollback-safe replacement.

        Constructs the candidate handler fully before mutating any root or
        global state. On failure, the old handler, root level, and globals
        are restored.
        """
        global _bootstrap_handler, _active_handler, _active_handler_mode

        self.install_filter()

        # Determine candidate mode and construct it before any mutation.
        if log_file is not None:
            candidate = logging.FileHandler(str(log_file), encoding="utf-8")
            candidate.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s"))
            new_mode = "file"
        elif rich and _rich_terminal_available():
            candidate = _get_rich_log_handler_class()(show_level=show_level, show_time=show_time)
            new_mode = "rich"
        else:
            candidate = _get_plain_log_handler_class()(show_level=show_level, show_time=show_time)
            new_mode = "plain"

        root = logging.getLogger()
        old_level = root.level
        old_bootstrap = _bootstrap_handler
        old_active = _active_handler
        old_mode = _active_handler_mode

        with _handler_lock:
            try:
                root.addHandler(candidate)
                root.setLevel(level)

                # Remove old linktools handlers.
                for old in (old_bootstrap, old_active):
                    if old is not None:
                        root.removeHandler(old)

            except Exception:
                # Rollback: remove candidate, re-add old handlers, restore level.
                try:
                    root.removeHandler(candidate)
                except Exception:
                    pass
                try:
                    candidate.close()
                except Exception:
                    pass
                for old in (old_bootstrap, old_active):
                    if old is not None and old not in root.handlers:
                        root.addHandler(old)
                root.setLevel(old_level)
                raise

            # Success: close old handlers, commit globals.
            for old in (old_bootstrap, old_active):
                if old is not None:
                    try:
                        old.close()
                    except Exception:
                        pass
            _bootstrap_handler = None
            _active_handler = candidate
            _active_handler_mode = new_mode

        self.bridge_third_party()

    def bridge_third_party(self) -> None:
        logging.getLogger("paramiko").setLevel(logging.WARNING)
        logging.getLogger("ssh.channel").setLevel(logging.CRITICAL)

    def close(self) -> None:
        self.remove_filter()


_rich_checked: "bool | None" = None


def _is_rich_importable() -> bool:
    global _rich_checked
    if _rich_checked is None:
        try:
            import rich  # noqa: F401
            _rich_checked = True
        except ImportError:
            _rich_checked = False
    return _rich_checked


def _rich_terminal_available() -> bool:
    if not _is_rich_importable():
        return False
    try:
        from rich import get_console
        return get_console().is_terminal
    except Exception:
        return False
