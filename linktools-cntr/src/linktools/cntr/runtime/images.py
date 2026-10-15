#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Strict image preparation for the final Docker Compose model."""
import hashlib
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..container import ContainerError
from .compose import service_dependencies, service_owners
from .structured import StructuredCommandError

if TYPE_CHECKING:
    from collections.abc import Sequence, Mapping
    from typing import Any
    from ..container import BaseContainer
    from ..manager import ContainerManager


class ImagePreparationError(ContainerError):
    pass


@dataclass(frozen=True)
class LocalImage:
    image_id: str
    revision: "str | None"


@dataclass(frozen=True)
class ImagePlan:
    build: "tuple[str, ...]"
    pull: "tuple[str, ...]"
    targets: "tuple[str, ...]"
    images: "dict[str, LocalImage | None]"


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
    """Classifies, checks, and prepares selected services and their dependencies."""

    def __init__(self, manager: "ContainerManager"):
        self.manager = manager

    BUILD_LABEL = "io.linktools.cntr.build-revision"
    BUILD_ARG = "CNTR_BUILD_REVISION"

    def with_build_revisions(self, model: "dict[str, Any]", containers,
                             targets: "Sequence[str]") -> "dict[str, Any]":
        """Inject managed build metadata into the ephemeral resolved model."""
        owners = service_owners(containers)
        services = dict(model["services"])
        for name in targets:
            spec = services[name]
            if spec.get("build") is None:
                continue
            build = dict(spec["build"])
            build["pull"] = False
            owner, revision = self._build_owner(name, build, owners[name])
            if revision is not None:
                labels = dict(build.get("labels") or {})
                args = dict(build.get("args") or {})
                if self.BUILD_LABEL in labels or self.BUILD_ARG in args:
                    raise ImagePreparationError("Reserved build metadata in " + name)
                identity = {
                    "revision": revision, "dockerfile": owner.docker_file,
                    "build": build, "platform": spec.get("platform"),
                }
                digest = hashlib.sha256(json.dumps(identity, sort_keys=True,
                        separators=(",", ":")).encode("utf-8")).hexdigest()
                labels[self.BUILD_LABEL] = digest
                args[self.BUILD_ARG] = digest
                build["labels"], build["args"] = labels, args
            services[name] = dict(spec, build=build)
        return dict(model, services=services)

    def _build_owner(self, service: str, build: "Mapping[str, Any]",
                     owners: "Sequence[BaseContainer]") -> "tuple[BaseContainer, str | None]":
        claims = [(owner, owner.get_build_revision(service)) for owner in owners]
        managed = [(owner, revision) for owner, revision in claims if revision is not None]
        if any(not isinstance(revision, str) or not revision for _, revision in managed):
            raise ImagePreparationError("Invalid build revision for " + service)
        if len(managed) <= 1:
            return managed[0] if managed else claims[0]
        matches = [(owner, revision) for owner, revision in managed
                   if self._matches_build(owner.services[service].get("build"), build)]
        if len(matches) == 1:
            return matches[0]
        candidates = matches or managed
        if len({(revision, owner.docker_file) for owner, revision in candidates}) == 1:
            return candidates[0]
        raise ImagePreparationError("Ambiguous managed build owners for {}: {}".format(
            service, ", ".join(owner.name for owner, _ in candidates)))

    def _matches_build(self, declared: "Mapping[str, Any] | str | None",
                       resolved: "Mapping[str, Any]") -> bool:
        import os
        if isinstance(declared, str):
            declared = {"context": declared}
        if not isinstance(declared, dict):
            return False
        if not any(key in declared for key in ("context", "dockerfile", "dockerfile_inline")):
            return False

        def absolute(value: str, base: "str | None" = None) -> str:
            if "://" in value or value.startswith("git@"):
                return value
            if os.path.isabs(value):
                return os.path.normpath(value)
            if base is None:
                base = os.path.join(str(self.manager.data_path), "compose")
            return os.path.normpath(os.path.join(base, value))

        context = resolved.get("context", ".")
        if absolute(declared.get("context", ".")) != absolute(context):
            return False
        if "dockerfile_inline" in declared or "dockerfile_inline" in resolved:
            return declared.get("dockerfile_inline") == resolved.get("dockerfile_inline")
        return (absolute(declared.get("dockerfile", "Dockerfile"), absolute(context)) ==
                absolute(resolved.get("dockerfile", "Dockerfile"), absolute(context)))

    def inspect_images(self, images: "Sequence[str]") -> "dict[str, LocalImage | None]":
        """Read one local batch; absent images are distinct from inspection failures."""
        images = tuple(dict.fromkeys(images))
        if not images:
            return {}
        try:
            process = self.manager.runtime.create_docker_process(
                "image", "inspect", *images, capture_output=True)
            result = self.manager.structured_runner.execute_text(process, check=False)
        except (OSError, StructuredCommandError) as exc:
            raise ImagePreparationError("Unable to inspect local images: " + str(exc)) from exc
        missing = set()
        if not result.succeeded:
            errors = [line for line in (result.stderr or "").splitlines() if line.strip()]
            for line in errors:
                match = re.fullmatch(r"(?:error response from daemon:\s*|error:\s*)?no such image:\s*(\S.*)",
                                     line.strip(), re.IGNORECASE)
                if match is None:
                    raise ImagePreparationError("Unable to inspect local images: " + (result.stderr or "").strip())
                missing.add(match.group(1))
            if not errors:
                raise ImagePreparationError("Unable to inspect local images: " + (result.stderr or "").strip())
        try:
            payload = json.loads(result.stdout)
            if not isinstance(payload, list):
                raise ValueError("expected an image array")
            states = []
            for item in payload:
                image_id = item["Id"]
                config = item.get("Config")
                if config is None:
                    config = {}
                if not isinstance(config, dict):
                    raise ValueError("invalid image configuration")
                labels = config.get("Labels")
                if labels is None:
                    labels = {}
                if not isinstance(image_id, str) or not image_id or not isinstance(labels, dict):
                    raise ValueError("invalid image metadata")
                revision = labels.get(self.BUILD_LABEL)
                if revision is not None and not isinstance(revision, str):
                    raise ValueError("invalid build revision")
                for field in ("RepoTags", "RepoDigests"):
                    references = item.get(field)
                    if references is not None and (not isinstance(references, list) or
                            any(not isinstance(reference, str) or not reference for reference in references)):
                        raise ValueError("invalid image references")
                states.append(LocalImage(image_id, revision))
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise ImagePreparationError("Invalid local image inspection output") from exc
        if result.succeeded:
            if len(states) != len(images):
                raise ImagePreparationError("Incomplete local image inspection output")
            return dict(zip(images, states))
        if not states:
            return dict.fromkeys(images)
        # Docker emits successful objects in argument order but omits failures.
        # Exact native references identify partial results; unresolved aliases
        # are queried alone rather than guessed from shifted array positions.
        observed = {}
        for item, state in zip(payload, states):
            for reference in (state.image_id, *(item.get("RepoTags") or ()), *(item.get("RepoDigests") or ())):
                observed[reference] = state
        resolved = {}
        for image in images:
            if image in missing:
                resolved[image] = None
            elif image in observed:
                resolved[image] = observed[image]
            elif len(images) == 1:
                raise ImagePreparationError("Unexpected partial image inspection output")
            else:
                resolved.update(self.inspect_images((image,)))
        return resolved

    def image_id(self, image: str) -> str:
        state = self.inspect_images((image,))[image]
        if state is None:
            raise ImagePreparationError("Cannot resolve local image ID for " + image)
        return state.image_id

    def image_revision(self, image: str) -> "str | None":
        state = self.inspect_images((image,))[image]
        if state is None:
            raise ImagePreparationError("Cannot inspect build metadata for " + image)
        return state.revision

    def verify_builds(self, model: "dict[str, Any]", services: "Sequence[str]",
                      images: "dict[str, LocalImage | None] | None" = None) -> None:
        """Check build results against a freshly observed image batch."""
        if images is None:
            images = self.inspect_images(tuple(model["services"][name]["image"] for name in services))
        for name in services:
            spec = model["services"][name]
            image = spec["image"]
            state = images[image]
            if state is None:
                raise ImagePreparationError("Built image is missing: " + image)
            expected = (spec.get("build") or {}).get("labels", {}).get(self.BUILD_LABEL)
            if expected is not None and state.revision != expected:
                raise ImagePreparationError("Built image revision mismatch for " + name)

    def plan(self, model: "dict[str, Any]", services: "Sequence[str]" = (),
             force_pull: bool = False, refresh_services: "Sequence[str] | None" = None,
             include_dependencies: bool = True) -> ImagePlan:
        """Plan selected images, including dependencies unless explicitly excluded."""
        all_services = model["services"]
        if not services:
            targets = list(all_services)
        elif include_dependencies:
            targets = _dependencies(all_services, services)
        else:
            targets = list(dict.fromkeys(services))
            for name in targets:
                if name not in all_services:
                    raise ImagePreparationError("Unknown Compose service dependency: " + name)
        refreshing = set(targets if refresh_services is None and force_pull else (refresh_services or ()))
        if not force_pull:
            refreshing.clear()
        build, pull = [], []
        requested = tuple(all_services[name].get("image") for name in targets)
        if any(not isinstance(image, str) or not image.strip() for image in requested):
            raise ImagePreparationError("Selected service has no valid image")
        image_state = self.inspect_images(requested)
        pull_images, revisions, build_images = set(), {}, set()
        for name in targets:
            service = all_services[name]
            image = service.get("image")
            definition = service.get("build")
            has_build = definition is not None
            expected = (definition.get("labels") or {}).get(self.BUILD_LABEL) if has_build else None
            if has_build and image in revisions and revisions[image] != expected:
                raise ImagePreparationError("Conflicting build revisions for image " + image)
            if has_build:
                revisions[image] = expected
            state = image_state[image]
            exists = state is not None
            refresh = name in refreshing
            if has_build and (refresh or not exists or
                              (expected is not None and state.revision != expected)):
                if image not in build_images:
                    build.append(name)
                    build_images.add(image)
            elif not has_build and (refresh or not exists) and image not in pull_images:
                pull.append(name)
                pull_images.add(image)
        return ImagePlan(tuple(build), tuple(pull), tuple(targets), image_state)

    def image_exists(self, image: str) -> bool:
        return self.inspect_images((image,))[image] is not None
