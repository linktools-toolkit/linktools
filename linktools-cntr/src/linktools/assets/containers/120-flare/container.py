#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Flare container definition."""
import os
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from linktools.cntr import BaseContainer, Flare, FlareLink, Nginx, ContainerError
from linktools.cntr.integration import IntegrationConsumer, load_port_url
from linktools.core import ConfigField, LazyProvider
from linktools.decorator import cached_property
from linktools.errors import ConfigNotFoundError
from linktools.rich import prompt

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import Any
    from linktools.cntr import EventContext, Integrations
    from linktools.cntr.artifacts import GeneratedCandidate


class Container(BaseContainer):

    @cached_property
    def configs(self) -> "dict[str, Any]":
        return dict(
            # NGINX_WILDCARD_DOMAIN is owned by the nginx container (its own
            # `configs` declares the field's cast/provider/default); this
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
        return [
            Flare.bookmark(
                "Flare", "bookmark", load_port_url(self, "FLARE_PORT", https=False),
                category="container",
            ),
            Nginx.site(
                server_name=self.get_config_later("FLARE_DOMAIN"),
                proxy="http://flare:5005",
                auth=None if self.get_config("FLARE_AUTH_ENABLE") else False,
                auth_bypass=(r"\.(css|js)$",),
                auth_rule={"policy": "one_factor"} if self.get_config("FLARE_AUTH_ENABLE") else None,
            ),
        ]


class Consumer(IntegrationConsumer):
    """Own the builtin flare consumer without extending container hooks."""

    generated = True
    application_order = 200

    def _iter_links(self) -> "Iterator[FlareLink]":
        manager = self.container.manager
        snapshot = manager.integration_snapshot
        producers = sorted(
            (name for name, declarations in snapshot.items() if declarations),
            key=lambda name: manager.containers[name].order,
        )
        for name in producers:
            declarations = snapshot[name]
            for declaration in declarations:
                if declaration.consumer != "nginx":
                    continue
                link = manager.nginx_sites[(name, declaration.local_id)].expose
                if link is not None:
                    yield link
            for declaration in declarations:
                if declaration.consumer == "flare":
                    yield declaration

    def on_render(self, generation_id: str) -> "dict[str, str]":

        categories = OrderedDict()
        apps = {"links": []}

        for link in self._iter_links():
            if not isinstance(link, FlareLink):
                continue
            url = link.url
            if not url:
                continue
            category = link.category
            existing = categories.get(category.name)
            if existing is None:
                existing = (category, [])
                categories[category.name] = existing
            else:
                for field, label in (("desc", "description"), ("apps", "output area"), ("order", "order")):
                    if getattr(existing[0], field) != getattr(category, field):
                        raise ContainerError(
                            f"Conflicting {label} for Flare category {category.name!r}")
            value = {"name": link.name, "icon": link.icon, "link": url}
            if category.apps:
                apps["links"].append(dict(value, desc=link.desc))
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

    def on_validate(self, context: "EventContext", candidate: "GeneratedCandidate") -> None:
        group = self.container.get_config("DOCKER_GID", type=int)
        for name in ("apps.yml", "bookmarks.yml"):
            path = Path(candidate.path) / name
            yaml.safe_load(path.read_text())
            # The service's configured group needs read access; the host owner
            # retains access for content comparison and future rollback.
            if path.stat().st_gid != group:
                self.container.runtime.create_process("chgrp", str(group), str(path), privilege=True).check_call()
            path.chmod(0o640)

    def on_apply(self, context: "EventContext", candidate: "GeneratedCandidate",
                 services: "Iterable[str]") -> None:
        if "flare" not in services:
            return
        app = self.container.get_app_path("app")
        app.mkdir(parents=True, exist_ok=True)
        migrated = []
        try:
            for name in ("apps.yml", "bookmarks.yml"):
                path = app / name
                target = "../generated/current/" + name
                if path.is_symlink() and os.readlink(str(path)) == target:
                    continue
                backup = None
                if path.exists() or path.is_symlink():
                    backup = app / (name + ".pre-cntr")
                    if backup.exists() or backup.is_symlink():
                        raise ContainerError("Flare migration backup already exists for " + name)
                    path.rename(backup)
                migrated.append((path, backup))
                temporary = app / (name + ".cntr-link")
                if temporary.exists() or temporary.is_symlink():
                    temporary.unlink()
                temporary.symlink_to(target)
                os.replace(str(temporary), str(path))
            runner = self.container.manager.compose_runner
            recreate = candidate.changed or not runner.is_generation_current(context, "flare", candidate)
            runner.apply_service(context, "flare", recreate=recreate)
            runner.wait_service_running(context, "flare")
        except Exception:
            for path, backup in reversed(migrated):
                if path.is_symlink():
                    path.unlink()
                if backup is not None:
                    backup.rename(path)
            raise
