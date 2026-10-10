#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Strict image preparation for the final Docker Compose model."""
import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..container import ContainerError
from .compose import service_dependencies
from .structured import StructuredCommandError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any
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
        for dep, options in service_dependencies(services[name]).items():
            if options.get("required", True) is not False:
                visit(dep)
        result.append(name)

    for name in targets:
        visit(name)
    return result


class ImagePreparer:
    """Classifies, checks, and prepares exactly the service dependency closure."""

    def __init__(self, manager: "ContainerManager"):
        self.manager = manager

    BUILD_LABEL = "io.linktools.cntr.build-revision"
    BUILD_ARG = "CNTR_BUILD_REVISION"

    def with_build_revisions(self, model: "dict[str, Any]", containers,
                             targets: "Sequence[str]") -> "dict[str, Any]":
        """Inject managed build metadata into the ephemeral resolved model."""
        owners = {name: owner for owner in containers for name in owner.services}
        services = dict(model["services"])
        for name in targets:
            spec = services[name]
            if spec.get("build") is None:
                continue
            build = dict(spec["build"])
            build["pull"] = False
            revision = owners[name].get_build_revision(name)
            if revision is not None:
                if not isinstance(revision, str) or not revision:
                    raise ImagePreparationError("Invalid build revision for " + name)
                labels = dict(build.get("labels") or {})
                args = dict(build.get("args") or {})
                if self.BUILD_LABEL in labels or self.BUILD_ARG in args:
                    raise ImagePreparationError("Reserved build metadata in " + name)
                identity = {
                    "revision": revision, "dockerfile": owners[name].docker_file,
                    "build": build, "platform": spec.get("platform"),
                }
                digest = hashlib.sha256(json.dumps(identity, sort_keys=True,
                        separators=(",", ":")).encode("utf-8")).hexdigest()
                labels[self.BUILD_LABEL] = digest
                args[self.BUILD_ARG] = digest
                build["labels"], build["args"] = labels, args
            services[name] = dict(spec, build=build)
        return dict(model, services=services)

    def image_id(self, image: str) -> str:
        """Resolve a local image ID before application without querying registries."""
        process = self.manager.runtime.create_docker_process(
            "image", "inspect", "--format", "{{.Id}}", image, capture_output=True)
        try:
            result = self.manager.structured_runner.execute_text(process, check=False)
        except (OSError, StructuredCommandError) as exc:
            raise ImagePreparationError("Cannot inspect image {}: {}".format(image, exc)) from exc
        if not result.succeeded or not result.stdout.strip():
            raise ImagePreparationError("Cannot resolve local image ID for " + image)
        return result.stdout.strip()

    def image_revision(self, image: str) -> "str | None":
        """Read a local image label; never resolve or pull from a registry."""
        command = self.manager.runtime.create_docker_process(
            "image", "inspect", "--format", "{{json .Config.Labels}}", image,
            capture_output=True)
        try:
            result = self.manager.structured_runner.execute_text(command, check=False)
        except (OSError, StructuredCommandError) as exc:
            raise ImagePreparationError("Cannot inspect build metadata for {}: {}".format(
                image, exc)) from exc
        if not result.succeeded:
            raise ImagePreparationError("Cannot inspect build metadata for {}: {}".format(
                image, result.stderr.strip()))
        try:
            labels = json.loads(result.stdout)
        except (TypeError, ValueError) as exc:
            raise ImagePreparationError("Invalid image label output for " + image) from exc
        if labels is not None and not isinstance(labels, dict):
            raise ImagePreparationError("Invalid image labels for " + image)
        return (labels or {}).get(self.BUILD_LABEL)

    def verify_builds(self, model: "dict[str, Any]", services: "Sequence[str]") -> None:
        """Do not accept an incomplete or mislabeled build as successful."""
        for name in services:
            spec = model["services"][name]
            image = spec["image"]
            if not self.image_exists(image):
                raise ImagePreparationError("Built image is missing: " + image)
            expected = (spec.get("build") or {}).get("labels", {}).get(self.BUILD_LABEL)
            if expected is not None and self.image_revision(image) != expected:
                raise ImagePreparationError("Built image revision mismatch for " + name)

    def plan(self, model: "dict[str, Any]", services: "Sequence[str]" = (),
             force_pull: bool = False, refresh_services: "Sequence[str] | None" = None) -> ImagePlan:
        all_services = model["services"]
        targets = list(all_services) if not services else _dependencies(all_services, services)
        refreshing = set(targets if refresh_services is None and force_pull else (refresh_services or ()))
        if not force_pull:
            refreshing.clear()
        build, pull = [], []
        image_state, pull_images, revisions, build_images = {}, set(), {}, set()
        for name in targets:
            service = all_services[name]
            image = service.get("image")
            definition = service.get("build")
            has_build = definition is not None
            if not isinstance(image, str) or not image.strip():
                raise ImagePreparationError("Service {} has no valid image".format(name))
            expected = (definition.get("labels") or {}).get(self.BUILD_LABEL) if has_build else None
            if has_build and image in revisions and revisions[image] != expected:
                raise ImagePreparationError("Conflicting build revisions for image " + image)
            if has_build:
                revisions[image] = expected
            if image not in image_state:
                image_state[image] = self.image_exists(image)
            exists = image_state[image]
            refresh = name in refreshing
            if has_build and (refresh or not exists or
                              (expected is not None and self.image_revision(image) != expected)):
                if image not in build_images:
                    build.append(name)
                    build_images.add(image)
            elif not has_build and (refresh or not exists) and image not in pull_images:
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
