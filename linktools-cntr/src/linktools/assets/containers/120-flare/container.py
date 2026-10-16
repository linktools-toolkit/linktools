#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Flare container definition."""
import os
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from linktools.cntr import BaseContainer, ContainerError
from linktools.cntr.ext import endpoint, Authelia, Flare, Nginx, load_port_url
from linktools.core import ConfigField, ConfigLazy, ConfigRef, LazyProvider
from linktools.decorator import cached_property
from linktools.errors import ConfigNotFoundError
from linktools.rich import prompt

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import Any
    from linktools.cntr import OperationContext, Integrations


class Container(BaseContainer):

    @cached_property
    def fields(self) -> "dict[str, Any]":
        return dict(
            # NGINX_WILDCARD_DOMAIN is owned by the nginx container (its own
            # `fields` declares the field's cast/provider/default); this
            # container must not redeclare it with a different default.
            FLARE_TAG="latest",
            FLARE_DOMAIN=Nginx.domain(self, ""),
            FLARE_PORT=ConfigField(cast=int, default=5000),
            FLARE_AUTH_ENABLE=ConfigField(cast=bool, default=True),
            FLARE_LOGIN_ENABLE=ConfigField(cast=bool, default=False),
            FLARE_USER=ConfigField.chain(
                LazyProvider(lambda r: self._prompt_flare_user(r), cached=True),
                default="",
            ),
            FLARE_PASSWORD=ConfigField.chain(
                LazyProvider(lambda r: self._prompt_flare_password(r), cached=True),
                default="",
            ),
        )

    def _prompt_flare_user(self, r):
        # Raise (rather than return "") when login is disabled, so the
        # enclosing ChainProvider falls through to field.default="" without
        # ever persisting it -- a plain cached=True here would otherwise
        # permanently cache "" the first time this resolves while login
        # happens to be off, and never prompt again once it's enabled.
        if not r.get("FLARE_LOGIN_ENABLE"):
            raise ConfigNotFoundError("FLARE_LOGIN_ENABLE is disabled")
        return prompt("FLARE_USER", default="admin")

    def _prompt_flare_password(self, r):
        if not r.get("FLARE_LOGIN_ENABLE"):
            raise ConfigNotFoundError("FLARE_LOGIN_ENABLE is disabled")
        return prompt("FLARE_PASSWORD")

    @cached_property
    def integrations(self) -> "Integrations":
        return endpoint(
            self,
            domain=ConfigRef("FLARE_DOMAIN"),
            proxy="http://flare:5005",
            auth=ConfigLazy(lambda config: None if config.get("FLARE_AUTH_ENABLE") else False),
            auth_bypass=(r"\.(css|js)$",),
            extensions=[
                Flare.container(
                    "Flare",
                    "bookmark",
                    load_port_url(ConfigRef("FLARE_PORT"), https=False),
                ),
                Authelia.access(
                    policy="one_factor",
                ),
            ],
        )

    def _iter_links(self) -> "Iterator[tuple[BaseContainer, Flare]]":
        declarations = sorted(
            self.manager.iter_integrations("flare"),
            key=lambda entry: entry[0].order,
        )
        for producer, declaration in declarations:
            yield producer, declaration


    def on_starting(self, context: "OperationContext") -> None:
        import shutil
        import tempfile
        app = self.get_app_path("runtime-app")
        if not app.exists():
            self.get_app_path().mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=".flare-app-", dir=str(self.get_app_path())))
            try:
                legacy = self.get_app_path("app")
                if legacy.is_dir():
                    for source in legacy.iterdir():
                        if source.name in ("apps.yml", "bookmarks.yml"):
                            continue
                        destination = temporary / source.name
                        if source.is_symlink():
                            destination.symlink_to(os.readlink(str(source)))
                        elif source.is_dir():
                            shutil.copytree(str(source), str(destination), symlinks=True)
                        else:
                            shutil.copy2(str(source), str(destination))
                self.runtime.chown(temporary, self.user, recursive=True)
                self.runtime.chmod(temporary, 0o750)
                os.rename(str(temporary), str(app))
            except BaseException:
                shutil.rmtree(str(temporary))
                raise
        context.write_files(self, self._navigation_files(), mode=0o640,
                            group=self.get_config("DOCKER_GID", type=int))

    def on_check(self, context: "OperationContext") -> None:
        for name in ("apps.yml", "bookmarks.yml"):
            yaml.safe_load(context.file_path(self, name).read_text(encoding="utf-8"))

    def _navigation_files(self) -> "dict[str, str]":

        categories = OrderedDict()
        apps = {"links": []}

        for producer, link in self._iter_links():
            if not isinstance(link, Flare):
                continue
            url = link.resolve_url(producer.config)
            if not url:
                continue
            category = link.display_category
            existing = categories.get(category.name)
            if existing is None:
                existing = (category, [])
                categories[category.name] = existing
            else:
                for field, label in (("desc", "description"), ("apps", "output area"), ("order", "order")):
                    if getattr(existing[0], field) != getattr(category, field):
                        raise ContainerError(
                            f"Conflicting {label} for Flare category {category.name!r}")
            value = deepcopy(link.resolve_metadata(producer.config, include_desc=category.apps))
            value["link"] = url
            if category.apps:
                apps["links"].append(value)
            else:
                existing[1].append(dict(value, category=category.name))

        bookmarks = {"categories": [], "links": []}
        for category, links in sorted(categories.values(), key=lambda entry: entry[0].order):
            if category.apps:
                continue
            bookmarks["categories"].append({"id": category.name, "title": category.desc})
            bookmarks["links"].extend(links)
        return {
            "apps.yml": yaml.safe_dump(apps, allow_unicode=True),
            "bookmarks.yml": yaml.safe_dump(bookmarks, allow_unicode=True),
        }
