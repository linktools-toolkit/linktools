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

    def __init__(self, *, index_urls: "Iterable[str]" = None):
        self._index_urls = index_urls or []

    @abc.abstractmethod
    def get_packages(self, name: str, deps: str) -> "Iterable[str]":
        pass

    def get_index_urls(self) -> "Iterable[str]":
        return self._index_urls


class DevelopUpdater(Updater):

    def __init__(self, project_path: "PathType", max_depth: int = 2, **kwargs):
        super().__init__(**kwargs)
        self._project_path = project_path
        self._max_depth = max_depth

    def get_packages(self, name: str, deps: str) -> "Iterable[str]":
        path = self.get_project_url(self._project_path, self._max_depth)
        if not path:
            raise ModuleError(
                f"{self._project_path} does not appear to be a Python project: "
                f"neither 'setup.py' nor 'pyproject.toml' found."
            )
        return ["--editable", f"{path}{deps}"]

    @classmethod
    def get_project_url(cls, path: "PathType", max_depth: int) -> Optional[pathlib.Path]:
        result = pathlib.Path(path)
        for i in range(max(max_depth, 0) + 1):
            if result.is_dir():
                if (result / "pyproject.toml").exists() or (result / "setup.py").exists():
                    return result
            result = result.parent
        return None


class GitUpdater(Updater):

    def __init__(self, repository_url: str = None, **kwargs):
        super().__init__(**kwargs)
        self._repository_url = repository_url

    def get_packages(self, name: str, deps: str) -> "Iterable[str]":
        repository_url = self._repository_url
        if not repository_url:
            repository_url = self.get_repository_url(name)
        if not repository_url:
            raise ModuleError(f"{name} has no repository url")
        return ["--ignore-installed", f"{name}{deps}@git+{repository_url.strip()}"]

    @classmethod
    def get_repository_url(cls, name: str):
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_packages(self, name: str, deps: str) -> "Iterable[str]":
        return ["--upgrade", f"{name}{deps}"]


class BaseCapability(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @property
    def version(self) -> str:
        return ""

    @property
    @abc.abstractmethod
    def updater(self) -> "Updater":
        pass


class Capability(BaseCapability):

    @property
    def name(self) -> str:
        return metadata.__name__

    @property
    def version(self) -> str:
        return metadata.__version__

    @property
    def develop(self) -> bool:
        return metadata.__develop__

    @property
    def release(self) -> bool:
        return metadata.__release__

    @property
    def updater(self) -> "Updater":
        from . import environ

        return next(filter(None, (   # noqa
            self.develop and DevelopUpdater(environ.root_path),
            not self.release and GitUpdater(),
            PypiUpdater()
        )))

    @property
    def root_path(self) -> pathlib.Path:
        from . import environ

        return environ.root_path

    def get_asset_path(self, *names: str) -> pathlib.Path:
        return self.root_path.joinpath("assets", *names)
