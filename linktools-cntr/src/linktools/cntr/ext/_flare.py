#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Flare application, bookmark, and category declarations."""
from typing import TYPE_CHECKING

from linktools.types import MISSING
from ._base import Integration

if TYPE_CHECKING:
    from typing import Optional


class _FlareCategory:
    """A Flare display group with an explicit output area and bookmark order."""

    def __init__(self, name: str, desc: str, *, apps: bool = False, order: int = 100) -> None:
        self.name = name
        self.desc = desc
        self.apps = apps
        self.order = order

    def __call__(self, name: str, icon: str, desc: str, url: "str | None" = MISSING) -> "Flare":
        return Flare(self, name, icon, desc or name, url)


class Flare(Integration):
    consumer = "flare"

    def __init__(self, category: "_FlareCategory", name: str, icon: str, desc: str, url: "str | None" = MISSING) -> None:
        self.display_category = category
        self.name = name
        self.icon = icon
        self.desc = desc
        self._url = url

    @property
    def url(self) -> "str | None":
        if self._url is MISSING or not self._url:
            return None
        return str(self._url)

    def with_default_url(self, url: str) -> "Flare":
        """Bind only an omitted URL, preserving explicit empty/None values."""
        if self._url is not MISSING:
            return self
        return Flare(self.display_category, self.name, self.icon, self.desc, url)

    _public = _FlareCategory("public", "Public", apps=True)
    _bookmarks = {
        "private": _FlareCategory("private", "Private", order=10),
        "container": _FlareCategory("container", "Internal", order=20),
        "other": _FlareCategory("other", "Tools", order=30),
    }

    @classmethod
    def public(cls, name: str, icon: str, desc: str, url: "str | None" = MISSING) -> "Flare":
        """Declare an application with a description."""
        return cls._public(name, icon, desc, url)

    @classmethod
    def container(cls, name: str, icon: str, url: "str | None" = MISSING, *,
                  desc: "Optional[str]" = None) -> "Flare":
        """Declare a bookmark in the standard container group."""
        return cls.bookmark(name, icon, url, category="container", desc=desc)

    @classmethod
    def category(cls, name: str, desc: "Optional[str]" = None, *,
                 apps: bool = False, order: "Optional[int]" = None) -> _FlareCategory:
        """Select a standard bookmark group or declare a custom display group."""
        existing = cls._bookmarks.get(name)
        if existing is not None and desc is None and not apps and order is None:
            return existing
        return _FlareCategory(name, name if desc is None else desc,
                             apps=apps, order=100 if order is None else order)

    @classmethod
    def bookmark(cls, name: str, icon: str, url: "str | None" = MISSING, *,
                 category: "str | _FlareCategory" = "other", desc: "Optional[str]" = None) -> "Flare":
        """Declare a bookmark using a category ID or explicit display group."""
        if isinstance(category, str):
            category = cls.category(category)
        if not isinstance(category, _FlareCategory):
            raise TypeError("bookmark category must be a string or _FlareCategory")
        if category.apps:
            raise ValueError("bookmark category must use the bookmarks output area")
        return Flare(category, name, icon, desc or name, url)
