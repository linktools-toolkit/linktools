#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Strict image preparation for the final Docker Compose model."""
import hashlib
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..container import ContainerError, SourceContainer
from .compose import service_dependencies, service_owners
from .structured import StructuredCommandError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any
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

    def with_build_revisions(self, model: "dict[str, Any]", containers,
                             targets: "Sequence[str]") -> "dict[str, Any]":
        """Inject managed build metadata into the ephemeral resolved model."""
        from pathlib import Path
        from ..artifacts import docker_file_candidate

        owners = service_owners(containers)
        services = dict(model["services"])
        for name in targets:
            spec = services[name]
            if spec.get("build") is None:
                continue
            build = dict(spec["build"])
            build["pull"] = False
            labels = dict(build.get("labels") or {})
            if self.BUILD_LABEL in labels:
                raise ImagePreparationError("Reserved build metadata in " + name)
            context = build.get("context", ".")
            dockerfile = build.get("dockerfile_inline")
            if "://" not in context and not context.startswith("git@"):
                context = Path(context)
                if not context.is_absolute():
                    context = self.manager.data_path / "compose" / context
                source = next((owner for owner in owners[name]
                               if isinstance(owner, SourceContainer) and
                               owner.get_docker_context_path().absolute() == context.absolute()), None)
                # Keep an absent source pointer eligible for image reuse without
                # fetching source. A dangling pointer still identifies its archive.
                snapshot = next((path for path in (context, *context.parents)
                                 if path.name == "current" and path.is_symlink()), None) if source else None
                if source is not None and snapshot is None:
                    services[name] = dict(spec, build=build)
                    continue
                if dockerfile is None:
                    path = Path(build.get("dockerfile", "Dockerfile"))
                    if not path.is_absolute():
                        path = context / path
                    for owner in owners[name]:
                        candidate = docker_file_candidate(owner)
                        if candidate is not None and candidate[0].resolve() == path.resolve():
                            dockerfile = candidate[1]
                    if dockerfile is None:
                        # Archive identity covers native files in source snapshots,
                        # including a tree that is currently absent from disk.
                        in_snapshot = False
                        if snapshot is not None:
                            try:
                                path.resolve().relative_to(snapshot.resolve())
                                in_snapshot = True
                            except ValueError:
                                pass
                        if not in_snapshot:
                            try:
                                dockerfile = path.read_text(encoding="utf-8")
                            except OSError as exc:
                                raise ImagePreparationError("Cannot read build Dockerfile for " + name) from exc
                context = str(context.resolve())
            identity = {
                "context": context, "dockerfile": dockerfile,
                "build": build, "platform": spec.get("platform"),
            }
            digest = hashlib.sha256(json.dumps(identity, sort_keys=True,
                    separators=(",", ":")).encode("utf-8")).hexdigest()
            labels[self.BUILD_LABEL] = digest
            build["labels"] = labels
            services[name] = dict(spec, build=build)
        return dict(model, services=services)

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
