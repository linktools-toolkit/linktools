#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only doctor checks (refactor spec Phase 6).

scan_compose is a pure function over a rendered compose dict; Doctor.run ties the
checks together and must never modify the config store.
"""
from linktools.cntr.doctor import Doctor, Finding, WARN, scan_compose


def _messages(findings):
    return [f.message for f in findings]


def test_scan_reports_latest_tag():
    assert any("latest" in m for m in _messages(scan_compose("c", {"services": {"a": {"image": "x:latest"}}})))


def test_scan_reports_untagged_image_as_latest():
    assert any("latest" in m for m in _messages(scan_compose("c", {"services": {"a": {"image": "nginx"}}})))


def test_scan_ignores_pinned_tag():
    assert not any("latest" in m for m in _messages(scan_compose("c", {"services": {"a": {"image": "nginx:1.25"}}})))


def test_scan_keeps_registry_port_out_of_tag():
    # registry.example:5000/x:1.2 -> tag is 1.2, not "5000/x".
    assert not any("latest" in m for m in _messages(
        scan_compose("c", {"services": {"a": {"image": "registry.example:5000/x:1.2"}}})))


def test_scan_reports_docker_socket_mount():
    msgs = _messages(scan_compose("c", {"services": {"a": {
        "image": "x:1", "volumes": ["/var/run/docker.sock:/var/run/docker.sock"]}}}))
    assert any("docker socket" in m for m in msgs)


def test_scan_reports_tls_disabled_list_env():
    msgs = _messages(scan_compose("c", {"services": {"a": {
        "image": "x:1", "environment": ["NODE_TLS_REJECT_UNAUTHORIZED=0"]}}}))
    assert any("NODE_TLS_REJECT_UNAUTHORIZED" in m for m in msgs)


def test_scan_reports_tls_disabled_dict_env():
    msgs = _messages(scan_compose("c", {"services": {"a": {
        "image": "x:1", "environment": {"NODE_TLS_REJECT_UNAUTHORIZED": "0"}}}}))
    assert any("NODE_TLS_REJECT_UNAUTHORIZED" in m for m in msgs)


def test_scan_clean_compose_has_no_warnings():
    findings = [f for f in scan_compose("c", {"services": {"a": {
        "image": "x:1.2", "volumes": ["/data:/data"]}}}) if f.severity == WARN]
    assert findings == []


def test_doctor_runs_on_builtins(fresh_manager):
    findings = Doctor(fresh_manager).run()
    assert isinstance(findings, list)
    assert all(isinstance(f, Finding) for f in findings)
    assert findings  # at least the runtime finding


def test_doctor_does_not_write_to_config_store(fresh_manager, monkeypatch):
    store = fresh_manager.environ.config_store
    writes = []
    original_set = store.set

    def spy(*args, **kwargs):
        writes.append(args)
        return original_set(*args, **kwargs)

    monkeypatch.setattr(store, "set", spy)
    Doctor(fresh_manager).run()
    assert writes == [], f"doctor wrote to config store: {writes}"
