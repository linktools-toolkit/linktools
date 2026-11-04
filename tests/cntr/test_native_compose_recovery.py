#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression contracts for reconciled native providers and optional Compose edges."""
from types import SimpleNamespace

import pytest

from linktools.cntr._operations import ComposeOperations, ComposeSelection
from linktools.cntr.runtime.compose import ComposeRunner, order_services
from linktools.cntr.runtime.images import ImagePreparer


class Owner:
    dependencies = ()
    application_priority = 0
    bootstrap_services = ()

    def __init__(self, name, services, generation_services=()):
        self.name = name
        self.services = {service: {} for service in services}
        self.generation_services = generation_services or tuple(services)


def test_optional_compose_provider_not_added_to_start_or_build_scope():
    app = Owner("app", ("app",))
    metrics = Owner("metrics", ("metrics",))
    app.services["app"]["depends_on"] = {
        "metrics": {"condition": "service_healthy", "required": False}}
    project = (app, metrics)
    model = {"services": {"app": app.services["app"], "metrics": {"image": "metrics:latest"}}}
    selected = ComposeOperations(SimpleNamespace()).required_selection(
        ComposeSelection(project, (app,), ("app",), False), model)
    assert selected.services == ("app",)
    assert order_services(project, ("app",), model) == ("app",)
    manager = SimpleNamespace()
    planner = ImagePreparer(manager)
    planner.inspect_images = lambda images: {image: SimpleNamespace(revision=None) for image in images}
    full_model = {"services": {"app": dict(app.services["app"], image="app:latest"),
                               "metrics": {"image": "metrics:latest"}}}
    assert planner.plan(full_model, ("app",)).targets == ("app",)


def test_restart_saved_orphan_uses_saved_compose_model(tmp_path):
    from pathlib import Path
    import yaml

    calls = []
    saved = {"services": {"retired": {"image": "old:image"}}}

    def process(*args):
        path = args[args.index("--file") + 1]
        assert yaml.safe_load(Path(path).read_text()) == saved
        calls.append(args)
        return SimpleNamespace(check_call=lambda: 0)

    manager = SimpleNamespace(data_path=tmp_path, project_name="test",
                              runtime=SimpleNamespace(create_docker_process=process))
    runner = ComposeRunner(manager)
    context = SimpleNamespace(is_full_project=True, compose_model={"services": {"current": {"image": "new:image"}}})
    runner.restart_service(context, "retired", model=saved)
    assert calls[0][-3:] == ("restart", "--no-deps", "retired")


@pytest.mark.parametrize("separate_owner", [False, True])
@pytest.mark.parametrize("services", [("cache", "web"), ("web", "cache")])
def test_legacy_dependency_removed_from_current_owner_uses_saved_declaration(tmp_path, separate_owner, services):
    from pathlib import Path
    import yaml

    old = {"services": {"db": {"image": "db:old"}, "cache": {"image": "cache:old"},
                        "web": {"image": "web:old", "network_mode": "service:db"}}}
    manager = SimpleNamespace(data_path=tmp_path, project_name="test",
                              runtime=SimpleNamespace(create_docker_process=lambda *args, **kw: SimpleNamespace(args=args)))
    runner = ComposeRunner(manager)

    def resolve(process):
        result = {"services": {}}
        for index, arg in enumerate(process.args[:-1]):
            if arg == "--file":
                result["services"].update(yaml.safe_load(Path(process.args[index + 1]).read_text())["services"])
        return result

    runner._resolved_model = resolve
    saved = {"app.yml": yaml.safe_dump(old)}
    if separate_owner:
        saved = {"app.yml": yaml.safe_dump({"services": {"web": old["services"]["web"]}}),
                 "provider.yml": yaml.safe_dump({"services": {name: old["services"][name] for name in ("db", "cache")}})}
    saved["unrelated.yml"] = "invalid: ["
    context = SimpleNamespace(project_containers=(Owner("app", ("web", "cache") if not separate_owner else ("web",)),
                                          Owner("provider", ("cache",)) if separate_owner else Owner("empty", ())),
                              previous_compose_contents=saved,
                              service_models=SimpleNamespace(previous={}))
    assert yaml.safe_load(runner.saved_service_models(context, services)["web"]) == old
