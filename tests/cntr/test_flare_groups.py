#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Flare display defaults compose without consuming lazy declaration values."""
import pytest

from linktools.cntr.ext import Flare, Nginx
from linktools.runtime import lazy_load
from linktools.types import MISSING


def fail() -> None:
    raise AssertionError("Flare composition must not resolve lazy values")


def test_flare_factories_accept_partial_metadata_without_using_category_titles() -> None:
    custom = Flare.category("team", "Team title", order=5)
    application = Flare.category("dashboard", "Dashboard title", apps=True)
    for factory in (Flare.public, Flare.container, Flare.bookmark, custom, application):
        link = factory()
        assert link._name is link._icon is link._desc is link._url is MISSING
        assert link.url is None
        grouped, = Flare.links(link, name="App", icon="web")
        assert (grouped.name, grouped.icon, grouped.desc) == ("App", "web", "App")
        assert grouped.display_category is link.display_category
    assert (custom.name, custom.desc, custom.apps, custom.order) == ("team", "Team title", False, 5)
    assert (application.name, application.desc, application.apps) == ("dashboard", "Dashboard title", True)


def test_flare_links_copies_plain_declarations_in_order_without_changing_urls() -> None:
    url = lazy_load(fail)
    originals = (Flare.public(), Flare.container(url=url),
                 Flare.bookmark("Own name", "own-icon", "https://docs.test", desc="Own description"))
    links = Flare.links(*originals, name="Shared name", icon="shared-icon", desc="Shared description")
    assert type(links) is tuple
    assert len(links) == len(originals)
    for original, link in zip(originals, links):
        assert type(link) is Flare
        assert link is not original
        assert link.display_category is original.display_category
        assert link._url is original._url
    assert [(link.name, link.icon, link.desc) for link in links] == [
        ("Shared name", "shared-icon", "Shared description"),
        ("Shared name", "shared-icon", "Shared description"),
        ("Own name", "own-icon", "Own description"),
    ]
    assert originals[0]._name is originals[0]._icon is originals[0]._desc is MISSING
    assert originals[1]._name is originals[1]._icon is originals[1]._desc is MISSING


def test_flare_links_preserves_explicit_none_empty_and_lazy_metadata() -> None:
    value = lazy_load(fail)
    for explicit in (None, "", value):
        original = Flare.public(name=explicit, icon=explicit, desc=explicit, url=value)
        grouped, = Flare.links(original, name="Shared", icon="shared", desc="Shared description")
        assert grouped._name is grouped._icon is grouped._desc is explicit
        assert original._name is original._icon is original._desc is explicit
        assert grouped.name is grouped.icon is explicit
        assert grouped._url is value


def test_flare_links_defaults_each_metadata_field_independently() -> None:
    name, = Flare.links(Flare.public("Own"), name="Shared", icon="web", desc="Details")
    icon, = Flare.links(Flare.public(icon="own"), name="Shared", icon="web", desc="Details")
    desc, = Flare.links(Flare.public(desc="Own"), name="Shared", icon="web", desc="Details")
    assert (name.name, name.icon, name.desc) == ("Own", "web", "Details")
    assert (icon.name, icon.icon, icon.desc) == ("Shared", "own", "Details")
    assert (desc.name, desc.icon, desc.desc) == ("Shared", "web", "Own")


def test_flare_description_fallback_uses_effective_link_name() -> None:
    for desc in (MISSING, None, ""):
        partial = Flare.public(desc=desc)
        inherited, = Flare.links(partial, name="Inherited", icon="web")
        own, = Flare.links(Flare.public("Own", desc=desc), name="Shared", icon="web")
        assert inherited.desc == "Inherited"
        assert own.desc == "Own"
        assert partial._desc is desc
    for desc in (None, ""):
        explicit, = Flare.links(Flare.public("Own", desc=desc), name="Shared", icon="web", desc="Details")
        assert explicit.desc == "Own"


def test_flare_description_is_not_evaluated_until_property_consumption() -> None:
    reads = []

    def description() -> str:
        reads.append("desc")
        return ""

    lazy = lazy_load(description)
    original = Flare.public(desc=lazy)
    grouped, = Flare.links(original, name="App", icon="web", desc=lazy_load(fail))
    filled = grouped.with_base_url(lazy_load(fail))
    assert reads == []
    assert original._desc is grouped._desc is filled._desc is lazy
    assert filled.desc == "App"
    assert reads == ["desc"]


def test_flare_group_defaults_stay_lazy_and_are_reusable() -> None:
    value = lazy_load(fail)
    partial = Flare.public()
    first, = Flare.links(partial, name=value, icon=value, desc=value)
    second, = Flare.links(partial, name="Second", icon="web")
    assert first._name is first._icon is first._desc is value
    assert (second.name, second.icon, second.desc) == ("Second", "web", "Second")
    assert partial._name is partial._icon is partial._desc is MISSING
    assert Flare.links(name=value, icon=value, desc=value) == ()


@pytest.mark.parametrize("value", [None, {}, [], (), "link", Nginx.site("app.test"), Flare.category("team")])
def test_flare_links_rejects_non_flare_declarations(value: object) -> None:
    with pytest.raises(TypeError, match="only Flare declarations"):
        Flare.links(Flare.public(), value, name="App", icon="web")


def test_flare_partial_metadata_fails_clearly_when_consumed() -> None:
    missing_name = Flare.public(icon="web", url="https://app.test")
    missing_icon = Flare.public(name="App", url="https://app.test")
    with pytest.raises(ValueError, match="requires a name"):
        missing_name.name
    with pytest.raises(ValueError, match="requires an icon"):
        missing_icon.icon


def test_default_url_fills_only_omitted_application_urls() -> None:
    url = lazy_load(fail)
    for factory in (Flare.public, Flare.category("dashboard", apps=True)):
        original = factory()
        filled = original.with_base_url(url)
        assert filled is not original
        assert filled._url is url
        assert original._url is MISSING
        assert filled._name is filled._icon is filled._desc is MISSING
        assert filled.display_category is original.display_category
    for factory in (Flare.container, Flare.bookmark, Flare.category("public", "Bookmarks")):
        original = factory()
        filled = original.with_base_url(url)
        assert filled is not original
        assert filled._url is original._url is MISSING


def test_default_url_preserves_every_explicit_url_and_all_metadata() -> None:
    value = lazy_load(fail)
    for url in (None, "", "https://own.test", value):
        original = Flare.public(value, value, value, url)
        filled = original.with_base_url(lazy_load(fail))
        assert filled is not original
        assert original._url is url
        if url is not value:
            assert filled.url == (url or None)
        assert filled._name is filled._icon is filled._desc is value
        assert original._name is original._icon is original._desc is value


def test_flare_group_and_url_defaults_compose_in_either_order() -> None:
    original = Flare.public()
    grouped, = Flare.links(original, name="App", icon="web")
    group_first = grouped.with_base_url("https://app.test")
    url_first, = Flare.links(original.with_base_url("https://app.test"), name="App", icon="web")
    for result in (group_first, url_first):
        assert (result.name, result.icon, result.desc, result.url) == ("App", "web", "App", "https://app.test")
    assert original._name is original._icon is original._desc is original._url is MISSING
