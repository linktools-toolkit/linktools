#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Unified logging management and handler ownership.

Business modules obtain their logger through ``environ.get_logger(name)``; the
:class:`LoggingManager` owns everything else -- the secret-redaction filter,
thread-local log context, the two-phase bootstrap/configure lifecycle,
third-party logger bridging (paramiko), and the process-global linktools log
handler.
"""
import ast
import base64
import binascii
import contextlib
import datetime
import json
import logging
import os
import re
import threading
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING
from urllib.parse import unquote

if TYPE_CHECKING:
    from typing import Any, Callable, Iterator, Match, Pattern

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

_SECRET_NAME = (
    r"(?:authorization|set[_-]?cookie|cookie|token|password|passwd|pwd|"
    r"secret[_\s-]?key|secret|api[_\s-]?key|apikey|access[_\s-]?key|"
    r"signature|private[_\s-]?key|(?<![^\W_])(?:session(?:[_\s-]?(?:id|key|token))?|sid))"
)
_QUOTE = r"\\*[\"']"
_HEADER_SECRET = r"(?:[\w.-]+[-_])?" + _SECRET_NAME
_SECRET_VALUE = re.compile(
    r"(?i)(?:"
    # Header tuples have no colon immediately after the header name.
    r"[\[(]\s*(?:b)?" + _QUOTE + _HEADER_SECRET + _QUOTE + r"\s*,\s*|"
    r"(?P<sshpass>sshpass\s+(?:-p\s+|--password[=\s]))|"
    + _SECRET_NAME + r"(?P<quoted>" + _QUOTE + r")?\s*[:=]\s*)"
)
_HEADER_NAME = re.compile(r"(?i)" + _HEADER_SECRET)
_HEADER_RECORD = re.compile(
    r"(?i)" + _QUOTE + r"(?:name|key)" + _QUOTE + r"\s*:\s*(?:b)?"
    + _QUOTE + _HEADER_SECRET + _QUOTE
)
_QUERY_SECRET = re.compile(r"(?i)(?:" + _HEADER_SECRET + r"|key)$")
_QUERY_VALUE = re.compile(
    r"(?i)(?P<separator>(?P<plain>[?&])|%3f|%26)(?P<key>[^&\s#=\"'<>]*?)"
    r"(?P<equals>=|%3d)"
    r"(?P<value>(?(plain)[^&\s#\"'<>]*|(?:(?!%26|%23)[^&\s#\"'<>])*))"
)
_VALUE_QUOTE = re.compile(r"(?:b)?(\\*)(\"{3}|'{3}|[\"'])")
_BASIC_VALUE = re.compile(r"(?i)(\bBasic\s+)([A-Za-z0-9+/]+={0,2})(?![\w+/=])")
_PEM_NEWLINE = r"(?:\r?\n|\\+r?\\*n)"
_PEM_BODY_LINE = r"[A-Za-z0-9+/]+={0,2}(?=$|[\r\n\"']|\\)"

_BUILTIN_REDACTORS: "list[tuple[Pattern[str], str]]" = [
    (re.compile(r"(://)[^\s/\"'<>?#]*@"), r"\1***@"),
    (re.compile(
        r"(?i)(%3a%2f%2f)"
        r"(?:(?!%2f|%3f|%23)[^\s/\"'<>?#])*(@|%40)"), r"\1***\2"),
    (re.compile(
        r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----[\s\S]*?"
        r"-----END (?:[A-Z]+ )?PRIVATE KEY-----"), "***"),
    (re.compile(
        r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----[ \t]*"
        r"(?:" + _PEM_NEWLINE + r"(?:Proc-Type:[ \t]*\d+,[ \t]*ENCRYPTED|"
        r"DEK-Info:[ \t]*[A-Z0-9-]+,[ \t]*[A-F0-9]+))*"
        r"(?:(?:" + _PEM_NEWLINE + r"[ \t]*)*" + _PEM_BODY_LINE
        + r"(?:(?:" + _PEM_NEWLINE + r"[ \t]*)+" + _PEM_BODY_LINE + r")*)?", re.I), "***"),
    (re.compile(r"(?i)(\bBearer\s+)[A-Za-z0-9._~+/-]+=*"), r"\1***"),
    (re.compile(r"(?<![\w-])(?:sk-[A-Za-z0-9_-]{20,}|AIza[A-Za-z0-9_-]{20,})(?![\w-])"), "***"),
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
        self._patterns: "list[tuple[Pattern[str], str]]" = []
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
        text = self._redact_text(text)
        for regex, repl in self._patterns:
            text = regex.sub(repl, text)
        if self._secrets:
            text = self._redact_encoded_strings(text, registered=True)
        for secret in self._secrets:
            if secret:
                text = text.replace(secret, "***")
        return text

    def _redact_text(self, text: str) -> str:
        text = self._redact_encoded_strings(text)
        for regex, repl in _BUILTIN_REDACTORS:
            text = regex.sub(repl, text)
        text = _BASIC_VALUE.sub(self._redact_basic, text)
        text = _QUERY_VALUE.sub(self._redact_query, text)
        text = self._redact_header_records(text)
        return self._redact_values(text)

    def _redact_encoded_strings(self, text: str, registered: bool = False) -> str:
        if not registered and "{" not in text and "[" not in text:
            return text
        parts = []
        consumed = position = 0
        while position < len(text):
            opening = _VALUE_QUOTE.search(text, position)
            if opening is None:
                break
            start = opening.start()
            end = self._quoted_end(text, start)
            position = end
            if opening.group(1):
                continue
            literal = text[start:end]
            for truncated in (False, True):
                candidate = literal + opening.group(2) if truncated else literal
                try:
                    decoded = json.loads(candidate)
                    json_string = True
                except ValueError:
                    try:
                        decoded = ast.literal_eval(candidate)
                    except (SyntaxError, ValueError):
                        continue
                    json_string = False
                break
            else:
                continue
            binary = isinstance(decoded, bytes)
            value = decoded.decode("latin1") if binary else decoded
            if not isinstance(value, str) or (not registered and "{" not in value and "[" not in value):
                continue
            if registered:
                redacted = self._redact_encoded_strings(value, registered=True)
                for secret in self._secrets:
                    redacted = redacted.replace(secret, "***")
            else:
                redacted = self._redact_text(value)
            if redacted == value:
                continue
            if json_string:
                replacement = json.dumps(redacted, ensure_ascii=all(ord(char) < 128 for char in literal))
            else:
                replacement = repr(redacted.encode("latin1") if binary else redacted)
            if truncated:
                replacement = replacement[:-1]
            parts.extend((text[consumed:start], replacement))
            consumed = end
        parts.append(text[consumed:])
        return "".join(parts)

    @classmethod
    def _redact_basic(cls, match: "Match[str]") -> str:
        value = match.group(2)
        try:
            credentials = base64.b64decode(value + "=" * (-len(value) % 4), validate=True)
        except (ValueError, binascii.Error):
            return match.group()
        return match.group(1) + "***" if b":" in credentials else match.group()

    @classmethod
    def _redact_query(cls, match: "Match[str]") -> str:
        if _QUERY_SECRET.fullmatch(unquote(match.group("key"))):
            return match.group("separator") + match.group("key") + match.group("equals") + "***"
        return match.group()

    @classmethod
    def _quoted_end(cls, text: str, start: int) -> int:
        opening = _VALUE_QUOTE.match(text, start)
        if opening is None:
            return start
        slashes, quote = opening.groups()
        position = opening.end()
        while True:
            position = text.find(quote, position)
            if position < 0:
                newline = text.find("\n", opening.end())
                return newline if newline >= 0 else len(text)
            backslash = position
            while backslash > start and text[backslash - 1] == "\\":
                backslash -= 1
            # JSON embedded in a repr may escape both its delimiters and
            # literal quotes. Each encoding layer doubles the escape width.
            if (position - backslash) % (2 * (len(slashes) + 1)) == len(slashes):
                return position + len(quote)
            position += len(quote)

    @classmethod
    def _redact_header_records(cls, text: str) -> str:
        if _HEADER_RECORD.search(text) is None:
            return text
        spans = []
        pending = [(0, len(text))]
        while pending:
            position, limit = pending.pop()
            stack = []
            while position < limit:
                if not stack:
                    position = text.find("{", position, limit)
                    if position < 0:
                        break
                char = text[position]
                opening = _VALUE_QUOTE.match(text, position)
                if opening is not None:
                    end = min(cls._quoted_end(text, position), limit)
                    content_end = end - len("".join(opening.groups()))
                    value = text[opening.end():content_end]
                    after = end
                    while after < limit and text[after].isspace():
                        after += 1
                    record = stack[-1]
                    if record["kind"] == "{" and after < limit and text[after] == ":":
                        record["key"] = value.lower()
                    else:
                        if record["key"] in ("name", "key") and _HEADER_NAME.fullmatch(value):
                            record["secret"] = True
                        elif record["key"] == "value":
                            record["values"].append((position, end))
                        record["key"] = None
                        if "{" in value:
                            pending.append((opening.end(), content_end))
                    position = end
                    continue
                if char in "[{(":
                    parent_value = bool(stack and stack[-1]["key"] == "value")
                    if stack:
                        stack[-1]["key"] = None
                    stack.append({"kind": char, "start": position, "key": None,
                                  "secret": False, "values": [], "parent_value": parent_value})
                elif char in "]})":
                    record = stack.pop()
                    if record["secret"]:
                        spans.extend(record["values"])
                    if stack and record["parent_value"]:
                        stack[-1]["values"].append((record["start"], position + 1))
                elif char not in ":, \t\r\n" and stack[-1]["key"] is not None:
                    end = re.compile(r"[,}\]]").search(text, position, limit)
                    end = end.start() if end is not None else limit
                    if stack[-1]["key"] == "value":
                        stack[-1]["values"].append((position, end))
                    stack[-1]["key"] = None
                    position = end
                    continue
                position += 1
            # Interrupted diagnostics can end before the record closes.
            for index in range(len(stack) - 1, -1, -1):
                record = stack[index]
                if record["secret"]:
                    spans.extend(record["values"])
                if index and record["parent_value"]:
                    stack[index - 1]["values"].append((record["start"], limit))
        parts = []
        consumed = 0
        for start, end in sorted(spans):
            if start < consumed:
                continue
            opening = _VALUE_QUOTE.match(text, start)
            replacement = opening.group() + "***" + "".join(opening.groups()) if opening else '"***"'
            parts.extend((text[consumed:start], replacement))
            consumed = end
        parts.append(text[consumed:])
        return "".join(parts)

    @classmethod
    def _redact_values(cls, text: str) -> str:
        parts = []
        consumed = 0
        queries = iter(_QUERY_VALUE.finditer(text))
        query = next(queries, None)
        for match in _SECRET_VALUE.finditer(text):
            if match.start() < consumed:
                continue
            while query is not None and query.end("key") <= match.start():
                query = next(queries, None)
            if query is not None and query.start() <= match.start() < query.end("key"):
                continue
            start = match.end()
            opening = _VALUE_QUOTE.match(text, start)
            replacement = "***"
            if opening is not None:
                end = cls._quoted_end(text, start)
                replacement = opening.group() + replacement + "".join(opening.groups())
            elif start < len(text) and text[start] in "[{(":
                stack = [text[start]]
                end = start + 1
                while end < len(text) and stack:
                    quote = _VALUE_QUOTE.match(text, end)
                    if quote is not None:
                        end = cls._quoted_end(text, end)
                        continue
                    char = text[end]
                    if char in "[{(":
                        stack.append(char)
                    elif char in "]})":
                        stack.pop()
                    end += 1
                replacement = '"***"'
            else:
                # Without quotes, spaces can be part of the secret. Retain
                # that conservative rule, but respect structured separators.
                if match.group("sshpass"):
                    boundary_pattern = r"\s"
                else:
                    boundary_pattern = r"[,}\]\r\n]" if match.group("quoted") else r"[\r\n]"
                boundary = re.compile(boundary_pattern).search(text, start)
                end = boundary.start() if boundary is not None else len(text)
            parts.extend((text[consumed:start], replacement))
            consumed = end
        parts.append(text[consumed:])
        return "".join(parts)

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

    def _create_display_handler(self, mode: str, show_level: bool, show_time: bool) -> "_LogHandlerMixin":
        """Pure factory: create a display handler for the given mode."""
        if mode == "rich":
            return _get_rich_log_handler_class()(show_level=show_level, show_time=show_time)
        elif mode == "plain":
            return _get_plain_log_handler_class()(show_level=show_level, show_time=show_time)
        else:
            raise ValueError("unknown display handler mode: %r" % (mode,))

    def _replace_display_handler(self, *, show_level: "bool | None" = None, show_time: "bool | None" = None) -> None:
        """Rollback-safe display handler toggle, fully inside _handler_lock."""
        global _active_handler

        with _handler_lock:
            mode = _active_handler_mode
            if mode not in ("plain", "rich"):
                return

            old_handler = _active_handler
            if not isinstance(old_handler, _LogHandlerMixin):
                return

            target_show_level = old_handler.show_level if show_level is None else show_level
            target_show_time = old_handler.show_time if show_time is None else show_time

            if target_show_level == old_handler.show_level and target_show_time == old_handler.show_time:
                return

            candidate = self._create_display_handler(mode, target_show_level, target_show_time)

            root = logging.getLogger()
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
        self._replace_display_handler(show_time=value)

    def set_show_level(self, value: bool) -> None:
        self._replace_display_handler(show_level=value)

    # -- two-phase lifecycle -----------------------------------------------

    def bootstrap(self) -> None:
        global _bootstrap_handler
        if self._bootstrapped:
            return
        self.install_filter()
        with _handler_lock:
            if self._bootstrapped:
                return
            root = logging.getLogger()
            old_level = root.level
            old_bootstrap = _bootstrap_handler
            added_handler = None
            try:
                if not root.handlers:
                    handler = logging.StreamHandler()
                    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
                    root.addHandler(handler)
                    _bootstrap_handler = handler
                    added_handler = handler
                if root.level == logging.NOTSET:
                    root.setLevel(logging.WARNING)
            except Exception:
                if added_handler is not None:
                    root.removeHandler(added_handler)
                    added_handler.close()
                _bootstrap_handler = old_bootstrap
                root.setLevel(old_level)
                raise
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

        Candidate is constructed outside the lock (no dependency on current
        state). All old-state capture and root mutation happens inside
        _handler_lock. On failure, old handlers, root level, and globals
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
        with _handler_lock:
            old_level = root.level
            old_bootstrap = _bootstrap_handler
            old_active = _active_handler

            try:
                root.addHandler(candidate)
                root.setLevel(level)
                for old in (old_bootstrap, old_active):
                    if old is not None:
                        root.removeHandler(old)
            except Exception:
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
