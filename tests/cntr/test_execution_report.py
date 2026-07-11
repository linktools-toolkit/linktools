#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution Report: opt-in --report; default behavior
(no --report) must be completely unchanged, including the return value of
on_command_up/restart/down (never a non-int, non-None value that could
reach sys.exit())."""
import linktools.cntr.__main__ as cntr_main
import linktools.cntr.commands._shared as cntr_shared
from linktools.cntr.execution.report import ExecutionRecord, get_records
from linktools.cntr.lifecycle.dispatcher import LifecycleDispatcher
from linktools.cntr.lifecycle.hooks import HookRegistry


def _record(manager, monkeypatch):
    def fake(containers, *args, privilege=None, **kwargs):
        class _Proc:
            def check_call(self):
                return 0
        return _Proc()

    monkeypatch.setattr(manager.runtime, "create_docker_compose_process", fake)
    monkeypatch.setattr(LifecycleDispatcher, "_invoke_callback", lambda self, func, context=None: None)
    monkeypatch.setattr(HookRegistry, "call", lambda self, phase, context=None, reverse=False: None)


def test_on_command_up_return_value_is_none(monkeypatch, fresh_manager):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    _record(fresh_manager, monkeypatch)
    result = cntr_main.command.on_command_up(names=["portainer"], build=False, pull=False)
    assert result is None


def test_on_command_up_return_value_is_none_with_report(monkeypatch, fresh_manager):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    _record(fresh_manager, monkeypatch)
    result = cntr_main.command.on_command_up(names=["portainer"], build=False, pull=False, report=True)
    assert result is None


def test_up_records_build_and_up_phases(fresh_manager, monkeypatch):
    _record(fresh_manager, monkeypatch)
    context_holder = []
    real_make_context = fresh_manager.compose_operations._make_context

    def spy_make_context(commands, selection):
        context = real_make_context(commands, selection)
        context_holder.append(context)
        return context

    monkeypatch.setattr(fresh_manager.compose_operations, "_make_context", spy_make_context)

    fresh_manager.compose_operations.up(names=["portainer"], build=True, pull=False)

    records = get_records(context_holder[0])
    phases = [r.phase for r in records]
    assert phases == ["build", "up"]
    assert all(isinstance(r, ExecutionRecord) and r.success for r in records)


def test_down_records_failure_with_message(fresh_manager, monkeypatch):
    monkeypatch.setattr(LifecycleDispatcher, "_invoke_callback", lambda self, func, context=None: None)
    monkeypatch.setattr(HookRegistry, "call", lambda self, phase, context=None, reverse=False: None)

    def fail(containers, *args, privilege=None, **kwargs):
        class _Proc:
            def check_call(self):
                raise RuntimeError("compose down failed")
        return _Proc()

    monkeypatch.setattr(fresh_manager.runtime, "create_docker_compose_process", fail)

    context_holder = []
    real_make_context = fresh_manager.compose_operations._make_context

    def spy_make_context(commands, selection):
        context = real_make_context(commands, selection)
        context_holder.append(context)
        return context

    monkeypatch.setattr(fresh_manager.compose_operations, "_make_context", spy_make_context)

    try:
        fresh_manager.compose_operations.down(names=["portainer"])
    except RuntimeError:
        pass

    records = get_records(context_holder[0])
    assert len(records) == 1
    assert records[0].phase == "down"
    assert records[0].success is False
    assert "compose down failed" in records[0].message


def test_failure_diagnostic_is_logged_regardless_of_report_flag(fresh_manager, monkeypatch):
    """On failure, phase/container/command/duration/error
    summary must always be shown -- independent of --report."""
    monkeypatch.setattr(LifecycleDispatcher, "_invoke_callback", lambda self, func, context=None: None)
    monkeypatch.setattr(HookRegistry, "call", lambda self, phase, context=None, reverse=False: None)

    def fail(containers, *args, privilege=None, **kwargs):
        class _Proc:
            def check_call(self):
                raise RuntimeError("compose down failed")
        return _Proc()

    monkeypatch.setattr(fresh_manager.runtime, "create_docker_compose_process", fail)

    errors = []
    monkeypatch.setattr(fresh_manager.logger, "error", lambda msg: errors.append(msg))

    try:
        fresh_manager.compose_operations.down(names=["portainer"])  # report defaults to False
    except RuntimeError:
        pass

    assert len(errors) == 1
    assert "down" in errors[0]
    assert "compose down failed" in errors[0]


def test_build_command_proxy_secrets_are_redacted(fresh_manager, monkeypatch):
    monkeypatch.setattr(LifecycleDispatcher, "_invoke_callback", lambda self, func, context=None: None)
    monkeypatch.setattr(HookRegistry, "call", lambda self, phase, context=None, reverse=False: None)
    monkeypatch.setenv("http_proxy", "http://user:super-secret@proxy:8080")

    def fail(containers, *args, privilege=None, **kwargs):
        class _Proc:
            def check_call(self):
                raise RuntimeError("build failed")
        return _Proc()

    monkeypatch.setattr(fresh_manager.runtime, "create_docker_compose_process", fail)

    context_holder = []
    real_make_context = fresh_manager.compose_operations._make_context

    def spy_make_context(commands, selection):
        context = real_make_context(commands, selection)
        context_holder.append(context)
        return context

    monkeypatch.setattr(fresh_manager.compose_operations, "_make_context", spy_make_context)

    try:
        fresh_manager.compose_operations.up(names=["portainer"], build=True, pull=False)
    except RuntimeError:
        pass

    records = get_records(context_holder[0])
    build_record = next(r for r in records if r.phase == "build")
    assert "super-secret" not in " ".join(build_record.command)
    assert "http_proxy=***" in build_record.command


def test_report_flag_does_not_change_running_state_writes(fresh_manager, monkeypatch):
    """--report must be purely additive: state writes happen the same way
    with or without it."""
    _record(fresh_manager, monkeypatch)
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)

    cntr_main.command.on_command_up(names=["portainer"], build=False, pull=False, report=True)
    with_report = set(fresh_manager.running_state.get_persisted())

    fresh_manager.running_state._set([])
    cntr_main.command.on_command_up(names=["portainer"], build=False, pull=False, report=False)
    without_report = set(fresh_manager.running_state.get_persisted())

    assert with_report == without_report == {"portainer"}
