#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Image snapshots belong to an operation phase, never a process-wide cache."""
import json
from types import SimpleNamespace

import pytest

from linktools.cntr.errors import ContainerError
from linktools.cntr.runtime.images import ImagePreparer
from test_lifecycle_rebuild import setup_case


def image_case(tmp_path, count=10, buildable=False):
    specs = {"app" + str(i): {"image": "image" + str(i) + ":local"} for i in range(count)}
    if buildable:
        for spec in specs.values():
            spec["build"] = {"context": "/fixture"}
    manager = setup_case(tmp_path, [("app", specs)])
    owner = manager.containers["app"]
    owner.docker_file = "FROM fixture"
    manager.image_preparer = ImagePreparer(manager)
    model = manager.image_preparer.with_build_revisions(manager.model, (owner,), tuple(specs))
    metadata = {spec["image"]: {"Id": "sha256:" + spec["image"],
                               "Config": {"Labels": spec.get("build", {}).get("labels", {})}}
                for spec in model["services"].values()}
    calls = []
    manager.runtime = SimpleNamespace(create_docker_process=lambda *args, **kwargs: args)
    def execute(args, **kwargs):
        calls.append(args)
        assert args[:2] == ("image", "inspect")
        return SimpleNamespace(succeeded=True, stdout=json.dumps([metadata[name] for name in args[2:]]), stderr="")
    manager.structured_runner = SimpleNamespace(execute_text=execute)
    return manager, metadata, calls


@pytest.mark.parametrize("buildable", [False, True])
def test_existing_local_images_need_two_batches_independent_of_service_count(tmp_path, buildable):
    manager, metadata, calls = image_case(tmp_path, buildable=buildable)
    manager.compose_operations.up(["app"])
    assert calls == [("image", "inspect", *metadata)] * 2
    assert not any(event[0] == "images" for event in manager.events)


def test_pull_refreshes_the_snapshot_before_final_image_verification(tmp_path):
    manager, metadata, calls = image_case(tmp_path, count=2)
    old_pull = manager.compose_runner.pull
    def pull(context, services):
        old_pull(context, services)
        for value in metadata.values():
            value["Id"] += "-pulled"
    manager.compose_runner.pull = pull
    manager.compose_operations.up(["app"], pull=True)
    assert len(calls) == 3
    assert all(call == ("image", "inspect", *metadata) for call in calls)


def test_tag_drift_during_checks_is_rejected_before_apply(tmp_path):
    manager, metadata, calls = image_case(tmp_path, count=2)
    manager.containers["app"].on_check = lambda context: metadata["image1:local"].update(Id="sha256:drifted")
    with pytest.raises(ContainerError, match="Selected image changed before deployment: app1"):
        manager.compose_operations.up(["app"])
    assert len(calls) == 2
    assert not any(event[0] == "apply" for event in manager.events)
