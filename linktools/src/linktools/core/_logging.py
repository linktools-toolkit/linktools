#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Unified logging management (spec §3.2, §5.4-§5.8, §22.3).

Business modules obtain their logger through ``environ.get_logger(name)``; the
:class:`LoggingManager` owns everything else -- the secret-redaction filter,
thread-local log context, the two-phase bootstrap/configure lifecycle, and
third-party logger bridging (paramiko). Modules must never call
``logging.basicConfig``/``addHandler``/``logger.setLevel`` directly (§5.8):
level changes go through :meth:`LoggingManager.set_level`.

Redaction is on by default. The filter sits on the root logger, so every record
-- regardless of which handler ultimately renders it -- is scrubbed before it
leaves the process, and annotated with any active context fields.
"""
import contextlib
import logging
import re
import threading
from typing import Any, Dict, List, Optional, Pattern, Tuple, Union

__all__ = ["LoggingManager"]

# (compiled pattern, replacement) pairs applied in order to every log message.
# These cover the secret categories listed in spec §5.6 (LOG-003) without any
# registration. Patterns run URL-creds -> query params -> headers/kv -> cli.
_BUILTIN_REDACTORS = [
    # type: List[Tuple[Pattern[str], str]]
    # URL credentials: mask the whole userinfo up to the LAST '@' before the
    # path (the RFC userinfo terminator), so a password containing '@' does not
    # leak its tail. [^\s/]* stops at the path, bounding the match.
    (re.compile(r"(://)[^\s/]*@"), r"://***@"),
    # URL query parameters that carry secrets (?token=...&api_key=...).
    (re.compile(
        r"(?i)([?&](?:access[_-]?token|api[_-]?key|apikey|token|key|signature|"
        r"secret|password|passwd|pwd)=)([^&\s#]+)"), r"\1***"),
    # Sensitive headers / key=value assignments (Authorization, Cookie, *token,
    # passwords, api keys, signatures, private keys). The prefix is BOUNDED
    # ({0,24}) so a long no-match input cannot cause catastrophic backtracking,
    # and it includes space so multi-word keys like "api key=" / "client secret="
    # are recognised. Bare "key" is intentionally NOT matched here (it would
    # clobber "monkey=" etc.); the query-pattern above handles "?key=".
    (re.compile(
        r"(?i)((?:authorization|set[_-]?cookie|cookie|"
        r"[a-z0-9 _.-]{0,24}(?:token|password|passwd|pwd|secret[_\s-]?key|secret|"
        r"api[_\s-]?key|apikey|access[_\s-]?key|signature|private[_\s-]?key))"
        r"\s*[:=]\s*)([^\r\n]*)"), r"\1***"),
    # sshpass password on the command line.
    (re.compile(r"(?i)(sshpass\s+(?:-p\s+|--password[=\s]))(\S+)"), r"\1***"),
]


class _LocalContext(threading.local):
    """Per-thread context fields (spec §5.7). Each thread gets its own dict."""

    def __init__(self):
        # type: () -> None
        super().__init__()
        self.fields = {}  # type: Dict[str, Any]


class LoggingManager(object):
    """Owns redaction, context, and logger level policy for an Environment.

    Redaction is installed globally via :func:`logging.setLogRecordFactory`, so
    it covers every record from every logger (a ``logging.Filter`` on the root
    *logger* would miss records emitted by child loggers, because Python does
    not re-apply ancestor logger filters during propagation).
    """

    def __init__(self, environ=None):
        # type: (Optional[Any]) -> None
        self._environ = environ
        self._secrets = []  # type: List[str]
        self._patterns = list(_BUILTIN_REDACTORS)  # type: List[Tuple[Pattern[str], str]]
        self._local = _LocalContext()
        self._installed = False
        self._old_factory = None  # type: Optional[Any]
        self._factory = None  # type: Optional[Any]
        self._bootstrapped = False

    # -- redaction ----------------------------------------------------------

    def register_secret(self, value):
        # type: (Any) -> None
        """Register a literal secret to be masked wherever it appears."""
        if isinstance(value, str) and value:
            self._secrets.append(value)

    def register_redactor(self, pattern, repl="***"):
        # type: (Union[str, Pattern[str]], str) -> None
        """Register an additional redaction ``pattern`` (str or compiled)."""
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
        self._patterns.append((pattern, repl))

    def redact(self, text):
        # type: (Any) -> Any
        """Apply all redactors + registered secrets to ``text``."""
        if not isinstance(text, str):
            return text
        for regex, repl in self._patterns:
            text = regex.sub(repl, text)
        for secret in self._secrets:
            if secret:
                text = text.replace(secret, "***")
        return text

    # -- context (§5.7) -----------------------------------------------------

    def current_context(self):
        # type: () -> Dict[str, Any]
        """Return a copy of the calling thread's context fields."""
        return dict(self._local.fields)

    @contextlib.contextmanager
    def context(self, **fields):
        # type: (**Any) -> Any
        """Scope context fields for the current thread (spec §5.7).

        Recommended keys: task_id, command, package, tool, repository, device,
        container, remote_host.
        """
        saved = dict(self._local.fields)
        self._local.fields.update(fields)
        try:
            yield
        finally:
            self._local.fields = saved

    # -- redaction installation (record factory) ---------------------------

    def install_filter(self):
        # type: () -> None
        """Install global redaction + context annotation (idempotent)."""
        if self._installed:
            return
        self._old_factory = logging.getLogRecordFactory()
        manager = self

        def _factory(*args, **kwargs):
            record = manager._old_factory(*args, **kwargs)
            for key, value in manager.current_context().items():
                record.__dict__.setdefault(key, value)
            # Redact the FINAL formatted message, never the raw format string.
            # Redacting record.msg directly would turn "password=%s" into
            # "password=***" and then getMessage()'s %-format would raise
            # TypeError (not all args converted), dropping the record.
            try:
                message = record.getMessage()
            except Exception:
                message = record.msg if isinstance(record.msg, str) else None
            if isinstance(message, str):
                record.msg = manager.redact(message)
                record.args = ()
            return record

        logging.setLogRecordFactory(_factory)
        self._factory = _factory
        self._installed = True

    def remove_filter(self):
        # type: () -> None
        """Restore the previous record factory."""
        if self._installed and self._old_factory is not None:
            logging.setLogRecordFactory(self._old_factory)
            self._installed = False
            self._factory = None
            self._old_factory = None

    # -- logger access (§3.2) ----------------------------------------------

    def get_logger(self, name=None):
        # type: (Optional[str]) -> logging.Logger
        """Return a named logger and ensure redaction is active."""
        self.install_filter()
        if name is None:
            name = getattr(self._environ, "name", None) or "linktools"
        return logging.getLogger(name)

    def set_level(self, name, level):
        # type: (str, int) -> None
        """Set a logger's level (modules use this instead of logger.setLevel)."""
        logging.getLogger(name).setLevel(level)

    # -- two-phase lifecycle (§5.5) ----------------------------------------

    def bootstrap(self):
        # type: () -> None
        """Phase 1 -- stderr handler at WARNING, no file/rich, redaction on.

        Idempotent. Safe at Environment creation; the only side effect on the
        root logger is adding a plain stderr handler when none exists.
        """
        if self._bootstrapped:
            return
        root = logging.getLogger()
        if not root.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            root.addHandler(handler)
        if root.level == logging.NOTSET:
            root.setLevel(logging.WARNING)
        self.install_filter()
        self._bootstrapped = True

    def configure(self, level=logging.INFO, log_file=None, rich=True):
        # type: (int, Optional[str], bool) -> None
        """Phase 2 -- apply the configured level and third-party bridging.

        Rich-handler/file-rotation integration with ``rich.py`` lands in a
        follow-up; for now this owns the level, the redaction filter, and the
        paramiko bridge (§13.9).
        """
        logging.getLogger().setLevel(level)
        self.install_filter()
        self.bridge_third_party()

    def bridge_third_party(self):
        # type: () -> None
        """Quiet chatty third-party loggers (spec §13.9 / §5.8).

        Replaces the ``_channel_logger.setLevel(...)`` call that used to live in
        ``ssh.py``: the SSH module no longer touches logger levels directly.
        """
        logging.getLogger("paramiko").setLevel(logging.WARNING)
        # paramiko routes transport diagnostics through this named channel.
        logging.getLogger("ssh.channel").setLevel(logging.CRITICAL)

    def close(self):
        # type: () -> None
        """Detach the record factory; file-handler flushing comes with rotation."""
        self.remove_filter()
