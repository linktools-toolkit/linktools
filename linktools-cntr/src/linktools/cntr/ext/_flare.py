#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Flare application, bookmark, and category declarations."""
from copy import copy
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from linktools.decorator import cached_property
from linktools.types import MISSING
from linktools.runtime import lazy_load
from ._base import Integration
from ._urls import is_complete_url, resolve_url_reference

if TYPE_CHECKING:
    from typing import Optional
    from linktools.core import Config, ConfigValue


class _FlareCategory:
    """A Flare display group with an explicit output area and bookmark order."""

    def __init__(self, name: str, desc: str, *, apps: bool = False, order: int = 100) -> None:
        self.name = name
        self.desc = desc
        self.apps = apps
        self.order = order

    def __call__(self, name: "str | ConfigValue | None" = MISSING,
                 icon: "str | ConfigValue | None" = MISSING,
                 desc: "str | ConfigValue | None" = MISSING,
                 url: "str | ConfigValue | None" = MISSING) -> "Flare":
        return Flare(self, name, icon, desc, url)


class Flare(Integration):
    consumer = "flare"

    def __init__(self, category: "_FlareCategory", name: "str | ConfigValue | None" = MISSING,
                 icon: "str | ConfigValue | None" = MISSING,
                 desc: "str | ConfigValue | None" = MISSING,
                 url: "str | ConfigValue | None" = MISSING) -> None:
        self.display_category = category
        self._name = name
        self._icon = icon
        self._desc = desc
        self._url = url
        self._base_urls = ()

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
        value = self._legacy_url if self._base_urls else self._url
        if value is MISSING or not value:
            return None
        return str(value)

    @cached_property
    def _legacy_url(self) -> "str | None":
        """Keep cached proxy composition confined to context-free property access."""
        value = self._url
        for base_url in self._base_urls:
            value = lazy_load(resolve_url_reference, value, base_url)
        return value

    def resolve_url(self, config: "Config") -> "str | None":
        """Resolve the URL in its producer's config before binding references."""
        value = config.resolve(self._url)
        if value is MISSING or not value:
            return None
        value = str(value)
        for base_url in self._base_urls:
            try:
                relative = not urlsplit(value).scheme
            except ValueError:
                return None
            if value.startswith("{url}") or relative:
                value = resolve_url_reference(value, config.resolve(base_url))
            if not value:
                return None
        if not is_complete_url(value):
            return None
        return value

    def resolve_metadata(self, config: "Config", *, include_desc: bool = True) -> "dict[str, str | None]":
        """Resolve display fields only after a usable URL has been selected."""
        name = config.resolve(self._name)
        if name is MISSING:
            raise ValueError("Flare link requires a name; set it on the link or with Flare.links")
        icon = config.resolve(self._icon)
        if icon is MISSING:
            raise ValueError("Flare link requires an icon; set it on the link or with Flare.links")
        result = {"name": name, "icon": icon}
        if include_desc:
            desc = config.resolve(self._desc)
            result["desc"] = name if desc is MISSING or not desc else desc
        return result

    def with_base_url(self, base_url: "str | ConfigValue") -> "Flare":
        """Copy this link with lazily bound default or relative URLs."""
        result = copy(self)
        if self._url is MISSING:
            if self.display_category.apps:
                result._url = base_url
        elif self._url is not None:
            result._base_urls = self._base_urls + (base_url,)
            result.__dict__.pop("_legacy_url", None)
        return result

    _public = _FlareCategory("public", "Public", apps=True)
    _bookmarks = {
        "private": _FlareCategory("private", "Private", order=10),
        "container": _FlareCategory("container", "Internal", order=20),
        "other": _FlareCategory("other", "Tools", order=30),
    }

    @classmethod
    def links(cls, *links: "Flare", name: "str | ConfigValue | None", icon: "str | ConfigValue | None",
              desc: "str | ConfigValue | None" = None) -> "tuple[Flare, ...]":
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
    def public(cls, name: "str | ConfigValue | None" = MISSING,
               icon: "str | ConfigValue | None" = MISSING,
               desc: "str | ConfigValue | None" = MISSING,
               url: "str | ConfigValue | None" = MISSING) -> "Flare":
        """Declare an application with a description."""
        return cls._public(name, icon, desc, url)

    @classmethod
    def container(cls, name: "str | ConfigValue | None" = MISSING,
                  icon: "str | ConfigValue | None" = MISSING,
                  url: "str | ConfigValue | None" = MISSING, *,
                  desc: "str | ConfigValue | None" = MISSING) -> "Flare":
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
    def bookmark(cls, name: "str | ConfigValue | None" = MISSING,
                 icon: "str | ConfigValue | None" = MISSING,
                 url: "str | ConfigValue | None" = MISSING, *, category: "str | _FlareCategory" = "other",
                 desc: "str | ConfigValue | None" = MISSING) -> "Flare":
        """Declare a bookmark using a category ID or explicit display group."""
        if isinstance(category, str):
            category = cls.category(category)
        if not isinstance(category, _FlareCategory):
            raise TypeError("bookmark category must be a string or _FlareCategory")
        if category.apps:
            raise ValueError("bookmark category must use the bookmarks output area")
        return Flare(category, name, icon, desc, url)
