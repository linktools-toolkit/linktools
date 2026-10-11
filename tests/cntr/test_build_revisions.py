#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from types import SimpleNamespace

import pytest

from linktools.cntr.runtime.images import ImagePreparer, ImagePreparationError, LocalImage


class Owner:
    services = {"app": {}}
    docker_file = "FROM alpine:3.21\nCOPY app /app\n"

    def __init__(self, revision="r1"):
        self.revision = revision

    def get_build_revision(self, name):
        assert name == "app"
        return self.revision


def case(revision="r1"):
    obj = ImagePreparer(SimpleNamespace())
    model = {"services": {"app": {"image": "app:fixed", "build": {
        "context": "/tmp/build", "dockerfile": "/tmp/Dockerfile", "pull": True}}}}
    return obj, obj.with_build_revisions(model, (Owner(revision),), ("app",))


def test_auto_label_is_stable_and_normal_up_disables_implicit_build_pull():
    prep, model = case()
    spec = model["services"]["app"]["build"]
    assert spec["pull"] is False
    assert spec["labels"][prep.BUILD_LABEL] == spec["args"][prep.BUILD_ARG]
    assert spec["labels"][prep.BUILD_LABEL] == case()[1]["services"]["app"]["build"]["labels"][prep.BUILD_LABEL]
    assert case("r2")[1]["services"]["app"]["build"]["labels"][prep.BUILD_LABEL] != spec["labels"][prep.BUILD_LABEL]


@pytest.mark.parametrize("exists,matching,refresh,build_expected", [
    (True, True, False, False), (True, False, False, True),
    (False, False, False, True), (True, True, True, True),
])
def test_build_decision(exists, matching, refresh, build_expected):
    prep, model = case()
    expected = model["services"]["app"]["build"]["labels"][prep.BUILD_LABEL]
    prep.inspect_images = lambda images: {image: LocalImage("sha256:fixture", expected if matching else None)
                                           if exists else None for image in images}
    plan = prep.plan(model, ("app",), force_pull=refresh, refresh_services=("app",) if refresh else ())
    assert bool(plan.build) is build_expected
    assert not plan.pull


def test_transitive_refresh_does_not_refresh_unrelated_consumer():
    prep, model = case()
    model["services"]["consumer"] = {"image": "consumer:latest"}
    prep.inspect_images = lambda images: {image: LocalImage("sha256:fixture",
        model["services"]["app"]["build"]["labels"][prep.BUILD_LABEL]) for image in images}
    plan = prep.plan(model, ("app", "consumer"), force_pull=True, refresh_services=("app",))
    assert plan.build == ("app",)
    assert plan.pull == ()


def test_no_revision_keeps_existing_image_rule():
    prep, model = case(None)
    assert "labels" not in model["services"]["app"]["build"]
    prep.inspect_images = lambda images: {image: LocalImage("sha256:fixture", None) for image in images}
    assert not prep.plan(model, ("app",)).build


def test_mismatched_result_cannot_be_accepted():
    prep, model = case()
    prep.inspect_images = lambda images: {image: LocalImage("sha256:fixture", "wrong") for image in images}
    with pytest.raises(ImagePreparationError, match="revision mismatch"):
        prep.verify_builds(model, ("app",))


def test_manual_reserved_build_metadata_rejected():
    prep = ImagePreparer(SimpleNamespace())
    model = {"services": {"app": {"image": "app:fixed", "build": {
        "context": "/tmp", "labels": {prep.BUILD_LABEL: "manual"}}}}}
    with pytest.raises(ImagePreparationError, match="Reserved"):
        prep.with_build_revisions(model, (Owner(),), ("app",))
