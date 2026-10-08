#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Flare container definition."""
import os
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from linktools.core import ConfigField, LazyProvider
from linktools.decorator import cached_property
from linktools.cntr import BaseContainer, NginxSite, ContainerError
from linktools.cntr.container import ExposeLink
from linktools.errors import ConfigNotFoundError
from linktools.rich import prompt

if TYPE_CHECKING:
    from typing import Any
    from linktools.cntr.artifacts import GeneratedCandidate
    from collections.abc import Iterable
    from linktools.cntr import EventContext


class Container(BaseContainer):

    @property
    def config_sources(self) -> "Iterable[str]":
        return tuple(container.name for container in self.manager.installed_state.get())

    @cached_property
    def configs(self) -> "dict[str, Any]":
        return dict(
            # NGINX_WILDCARD_DOMAIN is owned by the nginx container (its own
            # `configs` declares the field's cast/provider/default); this
            # container must not redeclare it with a different default.
            FLARE_TAG="latest",
            FLARE_DOMAIN=self.get_nginx_domain(""),
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
    def integrations(self) -> "dict[str, dict[str, NginxSite]]":
        return {
            "nginx": {
                "web": NginxSite(
                    server_name=self.get_config_later("FLARE_DOMAIN"),
                    proxy="http://flare:5005",
                    auth=None if self.get_config("FLARE_AUTH_ENABLE") else False,
                    auth_bypass=(r"\.(css|js)$",),
                    auth_rule={"policy": "one_factor"} if self.get_config("FLARE_AUTH_ENABLE") else None,
                ),
            },
        }

    @cached_property
    def exposes(self) -> "Iterable[ExposeLink]":
        return [
            self.expose_container("Flare", "bookmark", "主页", self.load_port_url("FLARE_PORT", https=False)),
        ]

    @property
    def generated_config_path(self) -> "Path":
        return self.get_app_path("generated")

    def render_generated_config(self, generation_id: str) -> "dict[str, str]":

        categories = {}
        apps = []
        bookmarks = []

        for container in sorted(self.manager.installed_state.get(), key=lambda o: o.order):
            for expose in container.exposes:
                if not isinstance(expose, ExposeLink) or not expose.is_valid:
                    continue
                category = expose.category
                existing = categories.get(category.name)
                if existing is None:
                    existing = (category.desc, [])
                    categories[category.name] = existing
                elif existing[0] != category.desc:
                    raise ContainerError(
                        f"Conflicting description for category {category.name!r}")
                existing[1].append(expose)
                if category.name == "public":
                    apps.append(expose)
                bookmarks.append(expose)

        data = {"links": []}
        for app in apps:
            data["links"].append({
                "name": app.name,
                "desc": app.desc,
                "icon": app.icon,
                "link": app.url,
            })
        result = {"apps.yml": yaml.safe_dump(data, allow_unicode=True)}

        data = {"categories": [], "links": []}
        for name, (description, links) in categories.items():
            if name == "public":
                continue
            if not links:
                continue
            data["categories"].append({
                "id": name,
                "title": description,
            })
            for link in links:
                data["links"].append({
                    "category": name,
                    "name": link.name,
                    "icon": link.icon,
                    "link": link.url,
                })
        result["bookmarks.yml"] = yaml.safe_dump(data, allow_unicode=True)
        return result

    def validate_generated_config(self, candidate: "GeneratedCandidate", context: "EventContext") -> None:
        group = self.get_config("DOCKER_GID", type=int)
        for name in ("apps.yml", "bookmarks.yml"):
            path = Path(candidate.path) / name
            yaml.safe_load(path.read_text())
            # The service's configured group needs read access; the host owner
            # retains access for content comparison and future rollback.
            if path.stat().st_gid != group:
                self.runtime.create_process("chgrp", str(group), str(path), privilege=True).check_call()
            path.chmod(0o640)

    def apply_generated_config(self, candidate: "GeneratedCandidate", context: "EventContext") -> None:
        app = self.get_app_path("app")
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
            runner = self.manager.compose_runner
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
