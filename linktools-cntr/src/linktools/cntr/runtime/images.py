#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Strict image preparation for the final Docker Compose model."""
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..container import ContainerError
from .structured import StructuredCommandError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any
    from ..context import EventContext
    from ..manager import ContainerManager


class ImagePreparationError(ContainerError):
    pass


@dataclass(frozen=True)
class ImagePlan:
    build: "tuple[str, ...]"
    pull: "tuple[str, ...]"
    targets: "tuple[str, ...]"


def _dependencies(services, targets):
    result, seen = [], set()

    def visit(name: str) -> None:
        if name in seen:
            return
        if name not in services:
            raise ImagePreparationError(f"Unknown Compose service dependency: {name}")
        seen.add(name)
        depends = services[name].get("depends_on", ())
        if isinstance(depends, dict):
            depends = depends.keys()
        for dep in depends or ():
            visit(dep)
        result.append(name)

    for name in targets:
        visit(name)
    return result


class ImagePreparer:
    """Classifies, checks, and prepares exactly the service dependency closure."""

    def __init__(self, manager: "ContainerManager"):
        self.manager = manager

    def plan(self, model: "dict[str, Any]", services: "Sequence[str]" = (), force_pull: bool = False) -> ImagePlan:
        all_services = model["services"]
        targets = list(all_services) if not services else _dependencies(all_services, services)
        build, pull = [], []
        image_state = {}
        pull_images = set()
        for name in targets:
            service = all_services[name]
            image = service.get("image")
            has_build = service.get("build") is not None
            if not isinstance(image, str) or not image.strip():
                raise ImagePreparationError(f"Service `{name}` has no valid image")
            if image not in image_state:
                image_state[image] = self.image_exists(image)
            exists = image_state[image]
            if has_build and (force_pull or not exists):
                build.append(name)
            elif not has_build and (force_pull or not exists) and image not in pull_images:
                pull.append(name)
                pull_images.add(image)
        return ImagePlan(tuple(build), tuple(pull), tuple(targets))

    def image_exists(self, image: str) -> bool:
        try:
            process = self.manager.runtime.create_docker_process(
                "image", "inspect", image, capture_output=True)
            result = self.manager.structured_runner.execute_text(process, check=False)
        except (OSError, StructuredCommandError) as exc:
            message = str(exc).lower()
            if "no such image" in message or "not found" in message:
                return False
            raise ImagePreparationError(f"Unable to inspect image `{image}`: {exc}") from exc
        if result.succeeded:
            return True
        message = (result.stderr or "").lower()
        if "no such image" in message or "not found" in message:
            return False
        raise ImagePreparationError(f"Unable to inspect image `{image}`: {result.stderr.strip()}")

    def execute(self, context: "EventContext", model: "dict[str, Any]", services: "Sequence[str]" = (), force_pull: bool = False) -> ImagePlan:
        plan = self.plan(model, services, force_pull=force_pull)
        if plan.pull:
            self.manager.compose_runner.pull(context, list(plan.pull))
        if plan.build:
            options = self.manager.compose_runner.options_for_build(plan.build, pull=force_pull)
            self.manager.compose_runner.build(context, options)
        return plan
