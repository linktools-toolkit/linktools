#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import abc
import pathlib
from typing import TYPE_CHECKING, Iterable, Optional

from .. import metadata
from ..types import ModuleError

if TYPE_CHECKING:
    from ..types import PathType


class Updater(metaclass=abc.ABCMeta):

    """Base updater interface for capability installation updates."""
    def __init__(self, *, index_urls: "Iterable[str]" = None):
        self._index_urls = index_urls or []

    @abc.abstractmethod
    def get_packages(self, name: str, deps: str) -> "Iterable[str]":
        """Return pip package specifiers needed for an update.

        Args:
            name (str): Name to resolve.
            deps (str): The deps value.

        Returns:
            Iterable[str]: The operation result.
        """
        pass

    def get_index_urls(self) -> "Iterable[str]":
        """Return extra package index URLs for installation.

        Returns:
            Iterable[str]: The operation result.
        """
        return self._index_urls


class DevelopUpdater(Updater):

    """Updater that operates against a local development checkout."""
    def __init__(self, project_path: "PathType", max_depth: int = 2, **kwargs):
        super().__init__(**kwargs)
        self._project_path = project_path
        self._max_depth = max_depth

    def get_packages(self, name: str, deps: str) -> "Iterable[str]":
        """Return editable package specifiers for a development checkout.

        Args:
            name (str): Name to resolve.
            deps (str): The deps value.

        Returns:
            Iterable[str]: The operation result.

        Raises:
            Exception: Propagates errors raised while completing the operation.
        """
        path = self.get_project_url(self._project_path, self._max_depth)
        if not path:
            raise ModuleError(
                f"{self._project_path} does not appear to be a Python project: "
                f"neither 'setup.py' nor 'pyproject.toml' found."
            )
        return ["--editable", f"{path}{deps}"]

    @classmethod
    def get_project_url(cls, path: "PathType", max_depth: int) -> Optional[pathlib.Path]:
        """Find a Python project directory near a path.

        Args:
            path (PathType): Filesystem path to process.
            max_depth (int): The max_depth value.

        Returns:
            Optional[pathlib.Path]: The operation result.
        """
        result = pathlib.Path(path)
        for i in range(max(max_depth, 0) + 1):
            if result.is_dir():
                if (result / "pyproject.toml").exists() or (result / "setup.py").exists():
                    return result
            result = result.parent
        return None


class GitUpdater(Updater):

    """Updater that installs capability code from a Git repository."""
    def __init__(self, repository_url: str = None, **kwargs):
        super().__init__(**kwargs)
        self._repository_url = repository_url

    def get_packages(self, name: str, deps: str) -> "Iterable[str]":
        """Return Git-backed package specifiers for installation.

        Args:
            name (str): Name to resolve.
            deps (str): The deps value.

        Returns:
            Iterable[str]: The operation result.

        Raises:
            Exception: Propagates errors raised while completing the operation.
        """
        repository_url = self._repository_url
        if not repository_url:
            repository_url = self.get_repository_url(name)
        if not repository_url:
            raise ModuleError(f"{name} has no repository url")
        return ["--ignore-installed", f"{name}{deps}@git+{repository_url.strip()}"]

    @classmethod
    def get_repository_url(cls, name: str):
        """Return the repository URL recorded in package metadata.

        Args:
            name (str): Name to resolve.

        Returns:
            Any: The operation result.
        """
        try:
            from importlib.metadata import distribution
        except ImportError:
            from importlib_metadata import distribution

        dist = distribution(name)
        for item in dist.metadata.get_all("Project-Url") or []:
            key, url = item.split(",", 1)
            if key.strip().lower() == "repository":
                return url.strip()
        return None


class PypiUpdater(Updater):

    """Updater that installs capability code from a Python package."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_packages(self, name: str, deps: str) -> "Iterable[str]":
        """Return package specifiers for a PyPI upgrade.

        Args:
            name (str): Name to resolve.
            deps (str): The deps value.

        Returns:
            Iterable[str]: The operation result.
        """
        return ["--upgrade", f"{name}{deps}"]


class BaseCapability(metaclass=abc.ABCMeta):
    """Base class for optional linktools capabilities."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return the name.

        Returns:
            str: The property value.
        """
        pass

    @property
    def version(self) -> str:
        """Return the version.

        Returns:
            str: The property value.
        """
        return ""

    @property
    @abc.abstractmethod
    def updater(self) -> "Updater":
        """Updater.

        Returns:
            Updater: The property value.
        """
        pass


class Capability(BaseCapability):
    """Default capability implementation for this package."""

    @property
    def name(self) -> str:
        """Return the name.

        Returns:
            str: The property value.
        """
        return metadata.__name__

    @property
    def version(self) -> str:
        """Return the version.

        Returns:
            str: The property value.
        """
        return metadata.__version__

    @property
    def develop(self) -> bool:
        """Develop.

        Returns:
            bool: The property value.
        """
        return metadata.__develop__

    @property
    def release(self) -> bool:
        """Release.

        Returns:
            bool: The property value.
        """
        return metadata.__release__

    @property
    def updater(self) -> "Updater":
        """Updater.

        Returns:
            Updater: The property value.
        """
        return next(filter(None, (   # noqa
            self.develop and DevelopUpdater(self.root_path),
            not self.release and GitUpdater(),
            PypiUpdater()
        )))

    @property
    def root_path(self) -> pathlib.Path:
        """Return the root path.

        Returns:
            pathlib.Path: The property value.
        """
        from . import environ

        return environ.root_path

    def get_asset_path(self, *names: str) -> pathlib.Path:
        """Return a path inside the capability assets directory.

        Args:
            names (str): Path or asset name components.

        Returns:
            pathlib.Path: The operation result.
        """
        return self.root_path.joinpath("assets", *names)
