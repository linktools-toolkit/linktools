#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Flare application, bookmark, and category declarations."""
from copy import copy
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

    def __call__(self, name: "str | None" = MISSING, icon: "str | None" = MISSING,
                 desc: "str | None" = MISSING, url: "str | None" = MISSING) -> "Flare":
        return Flare(self, name, icon, desc, url)


class Flare(Integration):
    consumer = "flare"

    def __init__(self, category: "_FlareCategory", name: "str | None" = MISSING,
                 icon: "str | None" = MISSING, desc: "str | None" = MISSING,
                 url: "str | None" = MISSING) -> None:
        self.display_category = category
        self._name = name
        self._icon = icon
        self._desc = desc
        self._url = url

    @property
    def name(self) -> "str | None":
        if self._name is MISSING:
            raise ValueError("Flare link requires a name; set it on the link or with Flare.links")
        return self._name

    @property
    def icon(self) -> "str | None":
        if self._icon is MISSING:
            raise ValueError("Flare link requires an icon; set it on the link or with Flare.links")
        return self._icon

    @property
    def desc(self) -> "str | None":
        return self.name if self._desc is MISSING or not self._desc else self._desc

    @property
    def url(self) -> "str | None":
        if self._url is MISSING or not self._url:
            return None
        return str(self._url)

    def with_default_url(self, url: "str | None") -> "Flare":
        """Copy this link, filling an omitted application URL without resolving it."""
        result = copy(self)
        if self.display_category.apps and self._url is MISSING:
            result._url = url
        return result

    _public = _FlareCategory("public", "Public", apps=True)
    _bookmarks = {
        "private": _FlareCategory("private", "Private", order=10),
        "container": _FlareCategory("container", "Internal", order=20),
        "other": _FlareCategory("other", "Tools", order=30),
    }

    @classmethod
    def links(cls, *links: "Flare", name: "str | None", icon: "str | None",
              desc: "str | None" = None) -> "tuple[Flare, ...]":
        """Copy links with shared defaults for omitted display metadata."""
        result = []
        for link in links:
            if not isinstance(link, Flare):
                raise TypeError("Flare.links accepts only Flare declarations")
            value = copy(link)
            if value._name is MISSING:
                value._name = name
            if value._icon is MISSING:
                value._icon = icon
            if value._desc is MISSING:
                value._desc = desc
            result.append(value)
        return tuple(result)

    @classmethod
    def public(cls, name: "str | None" = MISSING, icon: "str | None" = MISSING,
               desc: "str | None" = MISSING, url: "str | None" = MISSING) -> "Flare":
        """Declare an application with a description."""
        return cls._public(name, icon, desc, url)

    @classmethod
    def container(cls, name: "str | None" = MISSING, icon: "str | None" = MISSING,
                  url: "str | None" = MISSING, *, desc: "str | None" = MISSING) -> "Flare":
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
    def bookmark(cls, name: "str | None" = MISSING, icon: "str | None" = MISSING,
                 url: "str | None" = MISSING, *, category: "str | _FlareCategory" = "other",
                 desc: "str | None" = MISSING) -> "Flare":
        """Declare a bookmark using a category ID or explicit display group."""
        if isinstance(category, str):
            category = cls.category(category)
        if not isinstance(category, _FlareCategory):
            raise TypeError("bookmark category must be a string or _FlareCategory")
        if category.apps:
            raise ValueError("bookmark category must use the bookmarks output area")
        return Flare(category, name, icon, desc, url)
