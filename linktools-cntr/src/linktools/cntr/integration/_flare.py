#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Flare application, bookmark, and category declarations."""
from typing import TYPE_CHECKING

from linktools.types import MISSING
from ._base import Integration

if TYPE_CHECKING:
    from typing import Optional


class FlareCategory:
    """A Flare display group with an explicit output area and bookmark order."""

    def __init__(self, name: str, desc: str, *, apps: bool = False, order: int = 100) -> None:
        self.name = name
        self.desc = desc
        self.apps = apps
        self.order = order

    def __call__(self, name: str, icon: str, desc: str, url: "str | None" = MISSING) -> "FlareLink":
        return FlareLink(self, name, icon, desc or name, url)


class FlareLink(Integration):
    consumer = "flare"

    def __init__(self, category: "FlareCategory", name: str, icon: str, desc: str, url: "str | None" = MISSING) -> None:
        self.category = category
        self.name = name
        self.icon = icon
        self.desc = desc
        self._url = url

    @property
    def url(self) -> "str | None":
        if self._url is MISSING or not self._url:
            return None
        return str(self._url)

    def with_default_url(self, url: str) -> "FlareLink":
        """Bind only an omitted URL, preserving explicit empty/None values."""
        if self._url is not MISSING:
            return self
        return FlareLink(self.category, self.name, self.icon, self.desc, url)

    @property
    def is_valid(self) -> bool:
        return not not self.url


class Flare:
    """Factories for Flare applications, bookmarks, and display categories."""

    _public = FlareCategory("public", "Public", apps=True)
    _bookmarks = {
        "private": FlareCategory("private", "Private", order=10),
        "container": FlareCategory("container", "Internal", order=20),
        "other": FlareCategory("other", "Tools", order=30),
    }

    @classmethod
    def public(cls, name: str, icon: str, desc: str, url: "str | None" = MISSING) -> FlareLink:
        """Declare an application with a description."""
        return cls._public(name, icon, desc, url)

    @classmethod
    def category(cls, name: str, desc: "Optional[str]" = None, *,
                 apps: bool = False, order: "Optional[int]" = None) -> FlareCategory:
        """Select a standard bookmark group or declare a custom display group."""
        existing = cls._bookmarks.get(name)
        if existing is not None and desc is None and not apps and order is None:
            return existing
        return FlareCategory(name, name if desc is None else desc,
                             apps=apps, order=100 if order is None else order)

    @classmethod
    def bookmark(cls, name: str, icon: str, url: "str | None" = MISSING, *,
                 category: "str | FlareCategory" = "other") -> FlareLink:
        """Declare a bookmark using a category ID or explicit display group."""
        if isinstance(category, str):
            category = cls.category(category)
        if not isinstance(category, FlareCategory):
            raise TypeError("bookmark category must be a string or FlareCategory")
        if category.apps:
            raise ValueError("bookmark category must use the bookmarks output area")
        return FlareLink(category, name, icon, name, url)
