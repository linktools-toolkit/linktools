#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit image preparation leaves external dependency services untouched."""
from copy import deepcopy
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from linktools.cntr.runtime.images import ImagePreparationError, ImagePreparer

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any
    from linktools.cntr.runtime.images import LocalImage


def service_model() -> "dict[str, Any]":
    return {"services": {
        "safeline": {"image": "safeline:latest", "depends_on": {"nginx": {
            "condition": "service_started"}}},
        "sublink": {"image": "sublink:latest", "depends_on": ["nginx"]},
        "omniroute": {"image": "omniroute:latest", "depends_on": ["auth"]},
        "nginx": {"image": "nginx:managed", "build": {"context": "/tmp/nginx"},
                  "depends_on": ["auth"]},
        "auth": {"image": "auth:latest", "depends_on": ["lldap", "redis"]},
        "lldap": {"image": "lldap:latest"},
        "redis": {"image": "redis:latest"},
    }}


def missing_images() -> "tuple[ImagePreparer, list[tuple[str, ...]]]":
    preparer = ImagePreparer(SimpleNamespace())
    inspections = []

    def inspect(images: "Sequence[str]") -> "dict[str, LocalImage | None]":
        inspections.append(tuple(images))
        return dict.fromkeys(images)

    preparer.inspect_images = inspect
    return preparer, inspections


@pytest.mark.parametrize("services", [
    ("safeline",), ("sublink",), ("omniroute",),
    ("safeline", "sublink", "omniroute"),
])
@pytest.mark.parametrize("force_pull", [False, True])
def test_explicit_scope_does_not_inspect_pull_or_build_missing_dependencies(
        services: "tuple[str, ...]", force_pull: bool) -> None:
    preparer, inspections = missing_images()
    model = service_model()
    original = deepcopy(model)
    expected_images = tuple(model["services"][name]["image"] for name in services)

    plan = preparer.plan(model, services, force_pull=force_pull,
                         include_dependencies=False)

    assert plan.targets == services
    assert plan.pull == services
    assert plan.build == ()
    assert plan.images == dict.fromkeys(expected_images)
    assert inspections == [expected_images]
    assert model == original


def test_explicit_scope_builds_only_selected_services_even_with_broad_refresh() -> None:
    preparer, inspections = missing_images()
    model = service_model()
    model["services"]["safeline"]["build"] = {"context": "/tmp/safeline"}
    original = deepcopy(model)

    plan = preparer.plan(model, ("safeline", "omniroute"), force_pull=True,
                         refresh_services=tuple(model["services"]),
                         include_dependencies=False)

    assert plan.targets == ("safeline", "omniroute")
    assert plan.build == ("safeline",)
    assert plan.pull == ("omniroute",)
    assert inspections == [("safeline:latest", "omniroute:latest")]
    assert model == original


@pytest.mark.parametrize("options", [{}, {"include_dependencies": True}])
@pytest.mark.parametrize("force_pull", [False, True])
def test_default_scope_still_prepares_the_complete_dependency_closure(
        options: "dict[str, bool]", force_pull: bool) -> None:
    preparer, inspections = missing_images()
    model = service_model()
    original = deepcopy(model)

    plan = preparer.plan(model, ("safeline", "sublink", "omniroute"),
                         force_pull=force_pull, **options)

    assert plan.targets == ("lldap", "redis", "auth", "nginx", "safeline", "sublink", "omniroute")
    assert plan.build == ("nginx",)
    assert plan.pull == ("lldap", "redis", "auth", "safeline", "sublink", "omniroute")
    assert inspections == [tuple(model["services"][name]["image"] for name in plan.targets)]
    assert model == original


@pytest.mark.parametrize("include_dependencies", [False, True])
def test_empty_service_selection_still_prepares_the_full_model(include_dependencies: bool) -> None:
    preparer, inspections = missing_images()
    model = service_model()

    plan = preparer.plan(model, include_dependencies=include_dependencies)

    assert plan.targets == tuple(model["services"])
    assert plan.build == ("nginx",)
    assert plan.pull == ("safeline", "sublink", "omniroute", "auth", "lldap", "redis")
    assert inspections == [tuple(model["services"][name]["image"] for name in plan.targets)]


def test_explicit_scope_preserves_selected_order_and_deduplicates_services() -> None:
    preparer, inspections = missing_images()

    plan = preparer.plan(service_model(), ("omniroute", "safeline", "omniroute"),
                         include_dependencies=False)

    assert plan.targets == plan.pull == ("omniroute", "safeline")
    assert inspections == [("omniroute:latest", "safeline:latest")]


def test_explicit_scope_rejects_unknown_selection_before_inspecting_images() -> None:
    preparer, inspections = missing_images()

    with pytest.raises(ImagePreparationError, match="Unknown Compose service dependency: unknown"):
        preparer.plan(service_model(), ("unknown",), include_dependencies=False)

    assert inspections == []
