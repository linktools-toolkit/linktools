# -*- coding: utf-8 -*-
"""Tests for the linktools exception hierarchy (spec §18).

The hierarchy has a single root ``LinktoolsError`` with one direct child per
domain. ``Error`` is retained as an alias of ``LinktoolsError`` for the
migration cycle, so every historical name keeps working while ``isinstance``
checks against the new root succeed.
"""
import pytest

import linktools.errors as errs


DOMAIN_ROOTS = [
    "EnvironmentError", "LoggingError", "CacheError", "ConfigError",
    "DownloadError", "ToolError", "CapabilityError", "GitError", "SSHError",
    "SystemError", "ProcessError", "EventError", "ReactorError", "CliError",
    "ManifestError",
]

# Names that existed before this refactor and MUST stay importable so that
# linktools-common/mobile/cntr keep working without source changes.
LEGACY_NAMES = [
    "Error", "ModuleError", "DownloadError", "DownloadHttpError", "ExecError",
    "ConfigError", "ToolError", "ToolNotFound", "ToolNotSupport",
    "ToolExecError", "NoFreePortFoundError", "GitError", "GitDivergedError",
]


def test_error_is_alias_of_linktools_root():
    assert errs.Error is errs.LinktoolsError


def test_every_domain_root_is_a_linktools_error():
    for name in DOMAIN_ROOTS:
        cls = getattr(errs, name)
        assert issubclass(cls, errs.LinktoolsError), name


def test_all_legacy_names_still_importable():
    for name in LEGACY_NAMES:
        assert hasattr(errs, name), "lost legacy error name: %s" % name
        assert issubclass(getattr(errs, name), errs.LinktoolsError), name


def test_manifest_error_is_its_own_domain_root():
    """ManifestError must be a direct LinktoolsError child, not nested under
    ConfigError -- a manifest is a distinct core domain, not a config
    concern, and must not accidentally become catchable only via
    `except ConfigError`."""
    assert errs.ManifestError.__bases__ == (errs.LinktoolsError,)
    for name in ("ManifestLoadError", "ManifestValidationError", "ManifestRequirementError"):
        assert issubclass(getattr(errs, name), errs.ManifestError), name
    assert issubclass(errs.ManifestSchemaUnsupported, errs.ManifestValidationError)


def test_specifics_are_grouped_under_their_domain():
    assert issubclass(errs.ModuleError, errs.CapabilityError)
    assert issubclass(errs.ExecError, errs.ProcessError)
    assert issubclass(errs.NoFreePortFoundError, errs.SystemError)
    assert issubclass(errs.DownloadHttpError, errs.DownloadError)
    assert issubclass(errs.GitDivergedError, errs.GitError)
    for cls in (errs.ToolNotFound, errs.ToolNotSupport, errs.ToolExecError):
        assert issubclass(cls, errs.ToolError)


def test_root_catches_everything_domain_isolates_specifics():
    # Catching the root catches every domain.
    for cls in [errs.GitDivergedError, errs.ToolExecError, errs.ConfigError,
                errs.NoFreePortFoundError, errs.ModuleError]:
        assert isinstance(cls("x"), errs.LinktoolsError)

    # But a domain root only catches its own specifics.
    assert not isinstance(errs.GitError("x"), errs.ToolError)
    assert not isinstance(errs.ToolError("x"), errs.GitError)
    assert not isinstance(errs.ConfigError("x"), errs.CacheError)


def test_error_alias_catches_legacy_subclasses():
    # Code that does `except Error` (e.g. cntr ContainerError(Error)) must still
    # catch every linktools error after Error became the root alias.
    for cls in [errs.GitDivergedError, errs.ToolExecError, errs.DownloadError]:
        assert isinstance(cls("x"), errs.Error)


def test_download_http_error_carries_code():
    exc = errs.DownloadHttpError(503, "service unavailable")
    assert exc.code == 503


def test_all_public_names_listed_in___all__():
    # __all__ guards `from linktools.errors import *` so the builtin-shadowing
    # domain roots (SystemError, EnvironmentError) are opt-in only.
    assert set(errs.__all__) >= set(LEGACY_NAMES) | set(DOMAIN_ROOTS) | {"LinktoolsError"}


@pytest.mark.parametrize("name", DOMAIN_ROOTS + ["LinktoolsError"])
def test_str_does_not_crash(name):
    cls = getattr(errs, name)
    assert str(cls("boom"))
