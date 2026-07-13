"""Repository profile requirement checks owned by cntr."""

from typing import TYPE_CHECKING

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from linktools.errors import ConfigValidationError

if TYPE_CHECKING:
    from linktools.core import ProjectProfile


def ensure_requirement(profile: "ProjectProfile", name: str, actual_version: str) -> None:
    requires = profile.get("requires", {})
    if not isinstance(requires, dict):
        raise ConfigValidationError("'requires' must be an object")
    required = requires.get(name)
    if required is None:
        return
    if not isinstance(required, str) or not required.strip():
        raise ConfigValidationError("'requires.%s' must be a non-empty string" % name)
    try:
        specifier = SpecifierSet(required)
        version = Version(actual_version)
    except (InvalidSpecifier, InvalidVersion, TypeError) as exc:
        raise ConfigValidationError(
            "invalid requirement or version for `%s`: %s" % (name, exc)
        ) from exc
    if version not in specifier:
        raise ConfigValidationError(
            "`%s` requires %s, current version is %s" % (name, required, actual_version)
        )
