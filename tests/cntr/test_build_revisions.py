#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from types import SimpleNamespace

import pytest

from linktools.cntr import SourceContainer
from linktools.cntr.runtime.images import ImagePreparer, ImagePreparationError, LocalImage


class Owner(SourceContainer):
    def __init__(self, path, revision="r1"):
        self.path = path
        self._name = "source"
        self.__dict__["services"] = {"app": {}}
        self.__dict__["docker_file"] = None
        path.mkdir(parents=True, exist_ok=True)
        pointer = path / "current"
        if pointer.is_symlink():
            pointer.unlink()
        if revision is not None:
            tree = path / "versions" / revision
            tree.mkdir(parents=True, exist_ok=True)
            pointer.symlink_to("versions/" + revision)

    def get_docker_context_path(self):
        return self.path / "current"


def case(tmp_path, revision="r1"):
    owner = Owner(tmp_path / "source", revision)
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM alpine:3.21\nCOPY app /app\n")
    obj = ImagePreparer(SimpleNamespace())
    model = {"services": {"app": {"image": "app:fixed", "build": {
        "context": str(owner.get_docker_context_path()), "dockerfile": str(dockerfile), "pull": True}}}}
    return obj, obj.with_build_revisions(model, (owner,), ("app",))


def test_auto_label_is_stable_and_normal_up_disables_implicit_build_pull(tmp_path):
    prep, model = case(tmp_path)
    spec = model["services"]["app"]["build"]
    assert spec["pull"] is False
    assert "args" not in spec
    assert spec["labels"][prep.BUILD_LABEL] == case(tmp_path)[1]["services"]["app"]["build"]["labels"][prep.BUILD_LABEL]
    assert case(tmp_path, "r2")[1]["services"]["app"]["build"]["labels"][prep.BUILD_LABEL] != spec["labels"][prep.BUILD_LABEL]


@pytest.mark.parametrize("args", [None, {}, {"VERSION": "one"}, {"CNTR_BUILD_REVISION": "user-value"}])
def test_build_identity_preserves_user_args_and_original_digest(tmp_path, args):
    import hashlib
    import json

    owner = Owner(tmp_path / "source")
    prep = ImagePreparer(SimpleNamespace())
    build = {"context": str(owner.get_docker_context_path()), "dockerfile_inline": "FROM scratch"}
    if args is not None:
        build["args"] = dict(args)
    model = {"services": {"app": {"image": "app:fixed", "build": build}}}
    identity = {"context": str(owner.get_docker_context_path().resolve()),
                "dockerfile": "FROM scratch", "build": dict(build, pull=False), "platform": None}
    expected = hashlib.sha256(json.dumps(identity, sort_keys=True,
                                        separators=(",", ":")).encode("utf-8")).hexdigest()

    result = prep.with_build_revisions(model, (owner,), ("app",))["services"]["app"]["build"]
    assert result["labels"][prep.BUILD_LABEL] == expected
    assert result.get("args") == args
    assert build.get("args") == args
    assert "labels" not in build
    if args is not None:
        build["args"] = dict(args, VERSION="two")
        changed = prep.with_build_revisions(model, (owner,), ("app",))["services"]["app"]["build"]
        assert changed["args"] == build["args"]
        assert changed["labels"][prep.BUILD_LABEL] != expected


@pytest.mark.parametrize("exists,matching,refresh,build_expected", [
    (True, True, False, False), (True, False, False, True),
    (False, False, False, True), (True, True, True, True),
])
def test_build_decision(tmp_path, exists, matching, refresh, build_expected):
    prep, model = case(tmp_path)
    expected = model["services"]["app"]["build"]["labels"][prep.BUILD_LABEL]
    prep.inspect_images = lambda images: {image: LocalImage("sha256:fixture", expected if matching else None)
                                           if exists else None for image in images}
    plan = prep.plan(model, ("app",), force_pull=refresh, refresh_services=("app",) if refresh else ())
    assert bool(plan.build) is build_expected
    assert not plan.pull


def test_transitive_refresh_does_not_refresh_unrelated_consumer(tmp_path):
    prep, model = case(tmp_path)
    model["services"]["consumer"] = {"image": "consumer:latest"}
    prep.inspect_images = lambda images: {image: LocalImage("sha256:fixture",
        model["services"]["app"]["build"]["labels"][prep.BUILD_LABEL]) for image in images}
    plan = prep.plan(model, ("app", "consumer"), force_pull=True, refresh_services=("app",))
    assert plan.build == ("app",)
    assert plan.pull == ()


def test_no_revision_keeps_existing_image_rule(tmp_path):
    prep, model = case(tmp_path, None)
    assert "labels" not in model["services"]["app"]["build"]
    prep.inspect_images = lambda images: {image: LocalImage("sha256:fixture", None) for image in images}
    assert not prep.plan(model, ("app",)).build


def test_mismatched_result_cannot_be_accepted(tmp_path):
    prep, model = case(tmp_path)
    prep.inspect_images = lambda images: {image: LocalImage("sha256:fixture", "wrong") for image in images}
    with pytest.raises(ImagePreparationError, match="revision mismatch"):
        prep.verify_builds(model, ("app",))


def test_manual_reserved_build_metadata_rejected(tmp_path):
    prep, model = case(tmp_path)
    with pytest.raises(ImagePreparationError, match="Reserved"):
        prep.with_build_revisions(model, (Owner(tmp_path / "source"),), ("app",))


def test_missing_tree_keeps_identity_without_restoration(tmp_path):
    import shutil
    prep, model = case(tmp_path)
    expected = model["services"]["app"]["build"]["labels"][prep.BUILD_LABEL]
    build = dict(model["services"]["app"]["build"])
    build.pop("labels")
    owner = Owner(tmp_path / "source")
    shutil.rmtree(str(owner.get_docker_context_path().resolve()))
    result = prep.with_build_revisions({"services": {"app": {"image": "app:fixed", "build": build}}},
                                       (owner,), ("app",))
    assert result["services"]["app"]["build"]["labels"][prep.BUILD_LABEL] == expected
    prep.inspect_images = lambda images: {image: LocalImage("sha256:fixture", expected) for image in images}
    assert prep.plan(result).build == ()
    assert not owner.get_docker_context_path().exists()


def test_snapshot_dockerfile_does_not_require_restoring_missing_tree(tmp_path):
    import shutil
    owner = Owner(tmp_path / "source")
    prep = ImagePreparer(SimpleNamespace())
    model = {"services": {"app": {"image": "app:fixed", "build": {
        "context": str(owner.get_docker_context_path()), "dockerfile": "Dockerfile"}}}}
    path = owner.get_docker_context_path() / "Dockerfile"
    path.write_text("FROM scratch\n")
    before = prep.with_build_revisions(model, (owner,), ("app",))
    shutil.rmtree(str(owner.get_docker_context_path().resolve()))
    assert prep.with_build_revisions(model, (owner,), ("app",)) == before


def test_plain_build_context_keeps_existing_image_rule(tmp_path):
    prep = ImagePreparer(SimpleNamespace())
    owner = SimpleNamespace(services={"app": {}})
    model = {"services": {"app": {"image": "app:fixed", "build": {"context": str(tmp_path)}}}}
    result = prep.with_build_revisions(model, (owner,), ("app",))
    assert "labels" not in result["services"]["app"]["build"]


def test_external_dockerfile_change_changes_identity(tmp_path):
    prep, model = case(tmp_path)
    owner = Owner(tmp_path / "source")
    build = dict(model["services"]["app"]["build"])
    build.pop("labels")
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    result = prep.with_build_revisions({"services": {"app": {"image": "app:fixed", "build": build}}},
                                       (owner,), ("app",))
    assert result["services"]["app"]["build"]["labels"] != model["services"]["app"]["build"]["labels"]


@pytest.mark.parametrize("previous_file", [False, True])
def test_source_build_fingerprints_pending_generated_dockerfile(fresh_manager, tmp_path, previous_file):
    from linktools.cntr import OperationContext
    from test_source_revisions import Source

    root = tmp_path / "container"
    root.mkdir()
    template = root / "Dockerfile"
    template.write_text("FROM first\n")
    (root / "compose.yml").write_text("services:\n  app: {}\n")
    owner = Source(fresh_manager, root, name="source-build")
    digest = "a" * 64
    (owner._source_root / "versions" / digest / "pkg").mkdir(parents=True)
    owner._publish_source(digest)
    prep = ImagePreparer(fresh_manager)
    destination = owner.get_docker_file_destination()
    if previous_file:
        owner.get_docker_file_path()
        assert destination.read_text() == "FROM first"
    else:
        assert not destination.exists()
    model = {"services": {"app": dict(owner.services["app"])}}
    before = prep.with_build_revisions(model, (owner,), ("app",))
    template.write_text("FROM second\n")
    fresh_manager.lifecycle.prepare((owner,))
    after = prep.with_build_revisions(model, (owner,), ("app",))
    assert before != after
    if previous_file:
        assert destination.read_text() == "FROM first"
    else:
        assert not destination.exists()
    image = model["services"]["app"]["image"]
    previous = before["services"]["app"]["build"]["labels"][prep.BUILD_LABEL]
    prep.inspect_images = lambda images: {image: LocalImage("sha256:previous", previous) if previous_file else None}
    assert prep.plan(after).build == ("app",)
    context = OperationContext(actions=["up"], project_containers=[owner], target_containers=[owner],
                               target_services=("app",), compose_model=after)
    fresh_manager.runtime.create_docker_process = lambda *args, **kwargs: SimpleNamespace(check_call=lambda: 0)
    fresh_manager.compose_runner.build(context, fresh_manager.compose_runner.options_for_build(("app",)))
    assert destination.read_text() == "FROM second"
