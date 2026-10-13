#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Idempotent template-helper hook registration.

The mkdir/chown/chmod template helpers (and filters) must register start hooks
idempotently: rendering the same path twice -- within one template or across
compose + Dockerfile renders -- registers a hook only once. Distinct paths and
distinct actions are not collapsed. Default chown resolves the configured Docker IDs.

Tests measure the delta added by a render, since prepare_installed_containers
already registers some hooks (e.g. nginx-conf) via configs evaluation.
"""


from linktools.cntr.lifecycle import HookPhase

def _render(container, tmp_path, body):
    template = tmp_path / "t.j2"
    template.write_text(body, encoding="utf-8")
    container.render_template(template)


def _added(container, tmp_path, body):
    baseline = len(list(container.hooks.iter_phase(HookPhase.BEFORE_START)))
    _render(container, tmp_path, body)
    return len(list(container.hooks.iter_phase(HookPhase.BEFORE_START))) - baseline


def test_helpers_register_one_hook_each(fresh_manager, tmp_path):
    container = fresh_manager.containers["portainer"]
    assert _added(container, tmp_path,
                  "{{ mkdir('/a') }}{{ chmod('/a') }}{{ chown('/a', 'nobody') }}") == 3


def test_repeated_mkdir_same_path_is_deduped(fresh_manager, tmp_path):
    container = fresh_manager.containers["portainer"]
    assert _added(container, tmp_path, "{{ mkdir('/dup') }}{{ mkdir('/dup') }}{{ mkdir('/dup') }}") == 1


def test_distinct_paths_are_not_deduped(fresh_manager, tmp_path):
    container = fresh_manager.containers["portainer"]
    assert _added(container, tmp_path, "{{ mkdir('/a') }}{{ mkdir('/b') }}") == 2


def test_re_rendering_same_template_does_not_duplicate(fresh_manager, tmp_path):
    container = fresh_manager.containers["portainer"]
    _render(container, tmp_path, "{{ mkdir('/x') }}{{ chmod('/x') }}")
    after_first = len(list(container.hooks.iter_phase(HookPhase.BEFORE_START)))
    _render(container, tmp_path, "{{ mkdir('/x') }}{{ chmod('/x') }}")
    assert len(list(container.hooks.iter_phase(HookPhase.BEFORE_START))) == after_first


def test_chown_without_user_registers_configured_owner_once(fresh_manager, tmp_path):
    container = fresh_manager.containers["portainer"]
    assert _added(container, tmp_path, "{{ chown('/a') }}{{ chown('/a') }}") == 1


def test_helpers_available_as_filters(fresh_manager, tmp_path):
    container = fresh_manager.containers["portainer"]
    # `| mkdir | chmod` mirrors the builtin compose idiom (e.g. lldap, safeline).
    assert _added(container, tmp_path, "{{ ('/f') | mkdir | chmod }}") == 2


def test_shared_paths_prepare_on_selected_consumer_only(fresh_manager, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from linktools.cntr import BaseContainer, OperationContext
    from linktools.cntr.lifecycle import LifecycleDispatcher
    from linktools.cntr._container import template as template_module

    container = BaseContainer(fresh_manager, tmp_path, "consumer")
    shared = tmp_path / "shared-home"
    provider = SimpleNamespace(home_files={".client": shared})
    source = tmp_path / "shared.yml"
    source.write_text(
        "{% for name, path in provider.home_files.items() %}"
        "{{ path | mkdir | chown('1000:1000') }}:/workspace/{{ name }}"
        "{% endfor %}", encoding="utf-8",
    )
    events = []
    monkeypatch.setattr(template_module.os, "makedirs",
                        lambda path, **kwargs: events.append(("mkdir", str(path))))
    monkeypatch.setattr(fresh_manager.runtime, "chown",
                        lambda path, user, recursive=False: events.append(("chown", str(path), user, recursive)))
    first = container.render_template(source, provider=provider)
    assert first == str(shared) + ":/workspace/.client"
    assert container.render_template(source, provider=provider) == first
    assert events == []
    assert not shared.exists()
    assert len(list(container.hooks.iter_phase(HookPhase.BEFORE_START))) == 2
    context = OperationContext()
    context.target_containers = (container,)
    with LifecycleDispatcher(fresh_manager).notify_start(context):
        assert events == [("mkdir", str(shared)), ("chown", str(shared), "1000:1000", False)]
