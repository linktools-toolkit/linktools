#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Idempotent template-helper hook registration.

The mkdir/chown/chmod template helpers (and filters) must register start hooks
idempotently: rendering the same path twice -- within one template or across
compose + Dockerfile renders -- registers a hook only once. Distinct paths and
distinct actions are not collapsed. chown without a user registers nothing
(unchanged).

Tests measure the delta added by a render, since prepare_installed_containers
already registers some hooks (e.g. nginx-conf) via configs evaluation.
"""


def _render(container, tmp_path, body):
    template = tmp_path / "t.j2"
    template.write_text(body, encoding="utf-8")
    container.render_template(template)


def _added(container, tmp_path, body):
    baseline = len(container.start_hooks)
    _render(container, tmp_path, body)
    return len(container.start_hooks) - baseline


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
    after_first = len(container.start_hooks)
    _render(container, tmp_path, "{{ mkdir('/x') }}{{ chmod('/x') }}")
    assert len(container.start_hooks) == after_first


def test_chown_without_user_registers_nothing(fresh_manager, tmp_path):
    container = fresh_manager.containers["portainer"]
    assert _added(container, tmp_path, "{{ chown('/a') }}") == 0


def test_helpers_available_as_filters(fresh_manager, tmp_path):
    container = fresh_manager.containers["portainer"]
    # `| mkdir | chmod` mirrors the builtin compose idiom (e.g. lldap, safeline).
    assert _added(container, tmp_path, "{{ ('/f') | mkdir | chmod }}") == 2
