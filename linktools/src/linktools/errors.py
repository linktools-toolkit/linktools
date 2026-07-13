#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""linktools exception hierarchy (spec §18).

``LinktoolsError`` is the single root. Each direct child is a *domain root*
matching one subsystem. Specific exceptions live under their domain root so
callers can catch a whole subsystem without enumerating every case, while the
root catches everything::

    LinktoolsError
    +-- EnvironmentError      # environment / paths / composition root
    +-- LoggingError          # logging manager / handlers
    +-- CacheError            # local persistence (sqlite store)
    +-- ConfigError           # configuration sources / resolver
    +-- DownloadError         # url / transport / validators
    +-- ToolError             # managed external tools
    +-- CapabilityError       # installed-package capabilities
    +-- GitError              # dulwich-backed git operations
    +-- SSHError              # paramiko-backed ssh operations
    +-- SystemError           # platform / ports / network helpers
    +-- ProcessError          # subprocess runtime
    +-- EventError            # event bus
    +-- ReactorError          # scheduler
    +-- CliError              # command framework

``Error`` is retained as an alias of ``LinktoolsError`` for the in-repo
migration cycle (spec §3.3): existing subclasses such as ``ContainerError(Error)``
and legacy ``except Error`` sites keep working unchanged. New code should catch
``LinktoolsError`` or a domain root; ``Error`` will be retired once every
consumer has migrated.

Note: ``EnvironmentError`` and ``SystemError`` intentionally reuse names that
are also legacy Python builtins (aliases of ``OSError`` and the interpreter
``SystemError``). They are distinct classes here and are listed in ``__all__``
so ``from linktools.errors import *`` remains explicit and opt-in. No in-repo
code catches the builtin forms (verified at refactor time). See ADR-013.
"""

__all__ = [
    "LinktoolsError", "Error",
    # Domain roots.
    "EnvironmentError", "LoggingError", "CacheError", "ConfigError",
    "DownloadError", "ToolError", "CapabilityError", "GitError", "SSHError",
    "SystemError", "ProcessError", "EventError", "ReactorError", "CliError",
    # Specific exceptions.
    "ModuleError", "DownloadHttpError", "ExecError",
    "ToolNotFound", "ToolNotSupport", "ToolExecError",
    "NoFreePortFoundError", "GitDivergedError", "GitUnavailableError", "GitStashRestoreError",
    # Cache subtree (
    "CacheValueError", "CacheCodecError", "CacheTransactionError",
    "CacheBackendError", "CacheBusyError", "CacheCorruptedError",
    "CacheBackupError", "CacheRestoreError",
    # SSH subtree (
    "SSHConnectionError", "SSHAuthenticationError", "SSHHostKeyError",
    "SSHCommandError", "SSHChannelError", "SSHTransferError",
    "SSHForwardError", "SSHTimeoutError",
    # Runtime subtrees (
    "ProcessStartError", "ProcessTimeoutError", "ProcessCleanupError",
    # Config subtree (
    "ConfigFieldError", "ConfigNotFoundError", "ConfigCastError",
    "ConfigValidationError", "ConfigLoadError", "ConfigPersistenceError",
    "ConfigCycleError", "ConfigPromptError",
    # Tool subtree (
    "ToolDefinitionError", "ToolDependencyError", "ToolPlatformError",
    "ToolIntegrityError", "ToolArchiveError", "ToolInstallError",
    "ToolHealthcheckError",
]


class LinktoolsError(Exception):
    """Root of every linktools-specific exception.

    All linktools errors inherit from here, so ``except LinktoolsError`` is the
    broadest safe catch in business code. ``str(error)`` must never contain a
    secret; subclasses carry structured fields where useful.
    """


# Retained alias for the migration cycle (spec  Same class object, two
# names, so ``isinstance(x, Error)`` and ``isinstance(x, LinktoolsError)``
# agree and ``class X(Error)`` continues to work.
Error = LinktoolsError


# ---------------------------------------------------------------------------
# Domain roots (spec 
# ---------------------------------------------------------------------------

class EnvironmentError(LinktoolsError):
    """Environment, path resolution or composition-root failures."""


class LoggingError(LinktoolsError):
    """Logging manager, handler or formatter failures."""


class CacheError(LinktoolsError):
    """Local persistence (cache store) failures."""


class CacheValueError(CacheError):
    """A cache value or argument is invalid (e.g. a negative TTL)."""


class CacheCodecError(CacheError):
    """A value could not be (de)serialised by the configured codec."""


class CacheTransactionError(CacheError):
    """A cache transaction was used incorrectly (e.g. shared across threads)."""


class CacheBackendError(CacheError):
    """The cache backend (SQLite) is unavailable, locked or corrupted."""


class CacheBusyError(CacheError):
    """The cache backend is locked by another writer (busy timeout exceeded)."""


class CacheCorruptedError(CacheError):
    """The cache database failed an integrity check."""


class CacheBackupError(CacheError):
    """A cache backup operation failed."""


class CacheRestoreError(CacheError):
    """A cache restore operation failed."""


class ConfigError(LinktoolsError):
    """Configuration data is invalid or unavailable."""


class ConfigFieldError(ConfigError):
    """A config field definition is invalid."""


class ConfigNotFoundError(ConfigError):
    """A requested config key has no value and no default."""


class ConfigProviderUnavailable(ConfigNotFoundError):
    """A ChainProvider sub-provider has no value to offer; the chain should
    try its next sub-provider rather than treat this as a resolution
    failure. Anything else raised by a sub-provider (cast/validator/cycle
    errors, or a bare programming error) must propagate immediately instead
    of being swallowed as "just try the next one"."""


class ConfigCastError(ConfigError):
    """A config value could not be cast to the requested type."""


class ConfigValidationError(ConfigError):
    """A config value failed its validator."""


class ConfigLoadError(ConfigError):
    """A config file/source could not be loaded."""


class ConfigPersistenceError(ConfigError):
    """A persistent config write failed."""


class ConfigCycleError(ConfigError):
    """An Alias/Lazy dependency cycle was detected."""


class ConfigPromptError(ConfigError):
    """A config prompt could not be answered (non-interactive / cancelled)."""


class DownloadError(LinktoolsError):
    """A download failed."""


class ToolError(LinktoolsError):
    """Managed external-tool discovery or execution failed."""


class CapabilityError(LinktoolsError):
    """An installed-package capability is missing or unusable."""


class GitError(LinktoolsError):
    """A git repository operation failed."""


class SSHError(LinktoolsError):
    """An SSH operation failed."""


class SSHConnectionError(SSHError):
    """An SSH connection could not be established or was lost."""


class SSHAuthenticationError(SSHError):
    """SSH authentication failed."""


class SSHHostKeyError(SSHError):
    """A remote host key was missing, changed, or rejected."""


class SSHCommandError(SSHError):
    """A remote command exited with a failure status."""


class SSHChannelError(SSHError):
    """An SSH channel operation failed."""


class SSHTransferError(SSHError):
    """An SCP/SFTP file transfer failed."""


class SSHForwardError(SSHError):
    """A local or reverse port forward failed."""


class SSHTimeoutError(SSHError):
    """An SSH operation exceeded its timeout."""


class SystemError(LinktoolsError):
    """Platform, port or network helper failures."""


class ProcessError(LinktoolsError):
    """A subprocess operation failed."""


class EventError(LinktoolsError):
    """An event-bus operation failed."""


class ReactorError(LinktoolsError):
    """A scheduler operation failed."""


class CliError(LinktoolsError):
    """A CLI-framework operation failed."""


# ---------------------------------------------------------------------------
# Specific exceptions, grouped under their domain root.
# ---------------------------------------------------------------------------

class ModuleError(CapabilityError):
    """A linktools module/capability cannot be loaded or used."""


def missing_optional_class(name, extra, exc):
    """Return a placeholder class that raises ``ModuleError`` on instantiation.

    Used by packages that depend on an optional extra (e.g. ``linktools[git]``
    needs dulwich) so ``import linktools.git`` never fails on a missing dep --
    only actually *using* the class does. Defined once here (next to
    ``ModuleError``) so git/ssh/... share one implementation.
    """
    class _MissingOptional(object):
        def __init__(self, *args, **kwargs):
            raise ModuleError(
                "%s requires optional dependency extra `%s`: %s" % (name, extra, exc))
    _MissingOptional.__name__ = name
    return _MissingOptional


class DownloadHttpError(DownloadError):
    """A download failed with a specific HTTP status code.

    ``code`` carries the status; the message argument must already be
    secret-safe (URL redaction is applied by the download layer).
    """

    def __init__(self, code: int, e: object) -> None:
        super().__init__(e)
        self.code = code


class ExecError(ProcessError):
    """A subprocess failed to execute."""


class ProcessStartError(ProcessError):
    """A subprocess could not be started."""


class ProcessTimeoutError(ProcessError):
    """A subprocess exceeded its timeout."""


class ProcessCleanupError(ProcessError):
    """A subprocess could not be cleaned up (terminate/kill)."""


class ToolNotFound(ToolError):
    """A requested managed tool cannot be found."""


class ToolNotSupport(ToolError):
    """A managed tool is not supported in the current environment."""


class ToolExecError(ToolError):
    """A managed-tool process exited with an execution error."""


class ToolDefinitionError(ToolError):
    """A tool definition is malformed."""


class ToolDependencyError(ToolError):
    """A tool dependency is missing or cyclic."""


class ToolPlatformError(ToolError):
    """A tool is unavailable on the current platform/architecture."""


class ToolIntegrityError(ToolError):
    """A downloaded tool failed hash/size verification."""


class ToolArchiveError(ToolError):
    """A tool archive could not be extracted safely."""


class ToolInstallError(ToolError):
    """A tool installation failed."""


class ToolHealthcheckError(ToolError):
    """A tool failed its post-install healthcheck."""


class NoFreePortFoundError(SystemError):
    """No free TCP port could be obtained."""


class GitDivergedError(GitError):
    """A local branch has diverged from its remote and cannot fast-forward."""


class GitUnavailableError(GitError):
    """Git support is unavailable in this runtime."""


class GitStashRestoreError(GitError):
    """A STASH_AND_RESTORE sync's stash-pop failed after the sync itself
    either succeeded or failed -- the message summarizes both outcomes so
    the original sync failure (if any) is never silently replaced by the
    restore failure."""
