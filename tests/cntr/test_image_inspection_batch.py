#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch image inspection through the real structured runner and fake processes."""
import json
from types import SimpleNamespace

import pytest

from linktools.cntr.runtime.images import ImagePreparationError, ImagePreparer, LocalImage
from linktools.cntr.runtime.structured import StructuredCommandRunner


def metadata(image_id, tags=None, digests=None, revision=None):
    return {"Id": image_id, "RepoTags": tags, "RepoDigests": digests,
            "Config": {"Labels": {ImagePreparer.BUILD_LABEL: revision} if revision else None}}


def case(responses):
    calls, closed = [], []

    class Process:
        def __init__(self, args, response):
            self.args = args
            self.response = response
            self.returncode = response[2]

        def fetch(self, timeout=None):
            yield self.response[0].encode(), self.response[1].encode()

        def recursive_kill(self):
            pass

        def close(self):
            closed.append(self.args)

    def create(*args, **kwargs):
        assert kwargs == {"capture_output": True}
        calls.append(args)
        expected, payload, stderr, returncode = responses.pop(0)
        assert args == ("image", "inspect", *expected)
        stdout = payload if isinstance(payload, str) else json.dumps(payload)
        return Process(args, (stdout, stderr, returncode))

    manager = SimpleNamespace(runtime=SimpleNamespace(create_docker_process=create))
    manager.structured_runner = StructuredCommandRunner(manager)
    return ImagePreparer(manager), calls, closed


def test_empty_batch_does_not_create_a_process():
    preparer, calls, closed = case([])
    assert preparer.inspect_images(()) == {}
    assert calls == closed == []


def test_success_preserves_argument_order_and_deduplicates_requests():
    preparer, calls, closed = case([(("second", "first"), [
        metadata("sha256:second", revision="rev2"), metadata("sha256:first", revision="rev1")], "", 0)])
    result = preparer.inspect_images(("second", "first", "second"))
    assert list(result) == ["second", "first"]
    assert result == {"second": LocalImage("sha256:second", "rev2"), "first": LocalImage("sha256:first", "rev1")}
    assert len(calls) == 1 and closed == calls


def test_partial_batch_resolves_tags_digests_ids_and_only_retries_unmatched_alias():
    digest = "repo@sha256:" + "d" * 64
    image_id = "sha256:" + "a" * 64
    requests = ("tagged:latest", "missing:latest", digest, image_id, "short-alias")
    one = metadata("sha256:tagged", tags=["tagged:latest"], revision="r1")
    two = metadata("sha256:digested", digests=[digest], revision="r2")
    three = metadata(image_id)
    alias = metadata("sha256:aliased", tags=["canonical:latest"])
    preparer, calls, closed = case([
        (requests, [one, two, three, alias], "Error response from daemon: No such image: missing:latest", 1),
        (("short-alias",), [alias], "", 0),
    ])
    result = preparer.inspect_images(requests)
    assert result == {"tagged:latest": LocalImage("sha256:tagged", "r1"), "missing:latest": None,
                      digest: LocalImage("sha256:digested", "r2"), image_id: LocalImage(image_id, None),
                      "short-alias": LocalImage("sha256:aliased", None)}
    assert len(calls) == 2 and closed == calls


def test_partial_result_does_not_shift_later_image_onto_missing_argument():
    preparer, calls, _ = case([(("missing:latest", "found:latest"),
        [metadata("sha256:found", tags=["found:latest"])], "Error: No such image: missing:latest", 1)])
    assert preparer.inspect_images(("missing:latest", "found:latest")) == {
        "missing:latest": None, "found:latest": LocalImage("sha256:found", None)}
    assert len(calls) == 1


def test_alias_reported_with_normalized_missing_name_falls_back_without_guessing():
    found = metadata("sha256:found", tags=["found:latest"])
    preparer, calls, _ = case([
        (("found:latest", "missing"), [found], "Error: No such image: missing:latest", 1),
        (("missing",), [], "Error: No such image: missing:latest", 1),
    ])
    assert preparer.inspect_images(("found:latest", "missing"))["missing"] is None
    assert len(calls) == 2


def test_all_missing_images_are_one_query():
    preparer, calls, _ = case([(("one", "two"), [],
        "Error: No such image: one:latest\nError: No such image: two:latest\n", 1)])
    assert preparer.inspect_images(("one", "two")) == {"one": None, "two": None}
    assert len(calls) == 1


@pytest.mark.parametrize("stderr", [
    "permission denied while connecting to Docker", "Cannot connect to the Docker daemon",
    "Error: No such image: absent\npermission denied", "", "daemon failed: no such image metadata available",
])
def test_runtime_failures_are_not_treated_as_missing_images(stderr):
    preparer, calls, closed = case([(("app",), [], stderr, 1)])
    with pytest.raises(ImagePreparationError, match="Unable to inspect"):
        preparer.inspect_images(("app",))
    assert len(calls) == 1 and closed == calls


@pytest.mark.parametrize("payload", ["not-json", "", {}, [None], [{"Id": 123}],
                                        [{"Id": "sha256:app", "Config": {"Labels": []}}]])
def test_invalid_json_or_metadata_fails_closed(payload):
    preparer, calls, closed = case([(("app",), payload, "", 0)])
    with pytest.raises(ImagePreparationError, match="Invalid local image"):
        preparer.inspect_images(("app",))
    assert len(calls) == 1 and closed == calls


@pytest.mark.parametrize("field,value", [("RepoTags", "app"), ("RepoTags", [123]),
                                         ("RepoDigests", [{}]), ("RepoDigests", {"bad": "shape"})])
def test_malformed_native_references_fail_closed(field, value):
    record = metadata("sha256:app")
    record[field] = value
    preparer, calls, closed = case([(("app", "missing:latest"), [record],
        "Error: No such image: missing:latest", 1)])
    with pytest.raises(ImagePreparationError, match="Invalid local image"):
        preparer.inspect_images(("app", "missing:latest"))
    assert len(calls) == 1 and closed == calls


def test_successful_batch_must_have_one_object_per_unique_request():
    preparer, _, _ = case([(("first", "second"), [metadata("sha256:first")], "", 0)])
    with pytest.raises(ImagePreparationError, match="Incomplete"):
        preparer.inspect_images(("first", "second"))


def test_reported_missing_reference_wins_over_other_objects_stale_repo_tags():
    image_id = "sha256:" + "a" * 64
    preparer, calls, _ = case([((image_id, "removed:latest"),
        [metadata(image_id, tags=["removed:latest"])], "Error: No such image: removed:latest", 1)])
    assert preparer.inspect_images((image_id, "removed:latest")) == {
        image_id: LocalImage(image_id, None), "removed:latest": None}
    assert len(calls) == 1


def test_alias_fallback_permission_failure_does_not_become_absent():
    preparer, calls, closed = case([
        (("canonical:latest", "missing:latest", "alias"),
         [metadata("sha256:found", tags=["canonical:latest"])],
         "Error: No such image: missing:latest", 1),
        (("alias",), [], "permission denied", 1),
    ])
    with pytest.raises(ImagePreparationError, match="permission denied"):
        preparer.inspect_images(("canonical:latest", "missing:latest", "alias"))
    assert len(calls) == 2 and closed == calls


def test_missing_error_does_not_hide_truncated_json():
    preparer, _, _ = case([(("app",), "[", "Error: No such image: app:latest", 1)])
    with pytest.raises(ImagePreparationError, match="Invalid local image"):
        preparer.inspect_images(("app",))


def test_timeout_is_not_absence_and_process_is_closed():
    preparer, calls, closed = case([(("app",), [], "", None)])
    with pytest.raises(ImagePreparationError, match="timed out"):
        preparer.inspect_images(("app",))
    assert closed == calls


def test_process_creation_failure_is_reported_as_inspection_failure():
    def unavailable(*args, **kwargs):
        raise OSError("fixture process unavailable")

    preparer = ImagePreparer(SimpleNamespace(runtime=SimpleNamespace(create_docker_process=unavailable)))
    with pytest.raises(ImagePreparationError, match="fixture process unavailable"):
        preparer.inspect_images(("app",))
