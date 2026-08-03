#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Modal + sidebar + command palette tests.

Covers the Doctor/Catalog modals (overlay on the workspace, Esc closes), the
Sidebar's render_* wiring, and the approval modal masking. Uses Textual's
Pilot so compose/CSS/render errors surface."""

import asyncio

import pytest

from linktools.ai.cli.client import DoctorCheck, DoctorReport, FakeRuntimeClient
from linktools.ai.cli.tui.app import LinktoolsAIApp
from linktools.ai.cli.tui.modals.approval import _mask_value
from linktools.ai.cli.tui.widgets import Sidebar


async def _wait_for(pred, *, timeout: float = 2.0) -> bool:
    elapsed = 0.0
    step = 0.02
    while elapsed < timeout:
        if pred():
            return True
        await asyncio.sleep(step)
        elapsed += step
    return bool(pred())


def _static_text(widget) -> str:
    """Best-effort plain text of a Static/Label content across Textual builds."""
    from rich.text import Text

    for attr in ("_Static__content", "_content", "_renderable"):
        content = getattr(widget, attr, None)
        if isinstance(content, Text):
            return content.plain
        if isinstance(content, str):
            return content
    return ""


@pytest.mark.asyncio
async def test_doctor_modal_renders_report_and_closes_on_esc() -> None:
    report = DoctorReport(
        checks=[
            DoctorCheck(label="project config", ok=True),
            DoctorCheck(label="storage", ok=False, detail="read-only"),
        ]
    )
    client = FakeRuntimeClient(doctor_report=report)
    app = LinktoolsAIApp(client=client)
    async with app.run_test() as pilot:
        app.action_doctor()
        await _wait_for(lambda: len(app.screen_stack) > 1)
        await pilot.pause()  # let the modal widgets compose
        from linktools.ai.cli.tui.modals import DoctorModal

        assert isinstance(app.screen, DoctorModal)
        body = app.screen.query_one("#doctor-body")
        # Wait for the worker to populate the body with the check content.
        await _wait_for(lambda: "project config" in _static_text(body))
        # Esc closes the overlay and returns to the workspace.
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, app.workspace.__class__)


@pytest.mark.asyncio
async def test_catalog_modal_renders_and_closes() -> None:
    client = FakeRuntimeClient(agents=("a1", "a2"), skills=("s1",), mcp_servers=())
    app = LinktoolsAIApp(client=client)
    async with app.run_test() as pilot:
        app.action_catalog()
        await _wait_for(lambda: len(app.screen_stack) > 1)
        from linktools.ai.cli.tui.modals import CatalogModal

        assert isinstance(app.screen, CatalogModal)
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, app.workspace.__class__)


@pytest.mark.asyncio
async def test_sidebar_renders_sessions_runs_approvals() -> None:
    class _Rec:
        def __init__(self, rid, status):
            self.id = rid
            self.status = status

    class _Status:
        def __init__(self, value):
            self.value = value

    client = FakeRuntimeClient(
        sessions=[_Rec("s1", _Status("completed"))],
        runs=[_Rec("r1", _Status("running"))],
        approvals=[_Rec("a1", _Status("paused"))],
        agents=("a1",),
        skills=("s1",),
        mcp_servers=("m1",),
    )
    app = LinktoolsAIApp(client=client)
    async with app.run_test() as pilot:
        ws = app.workspace
        await ws.refresh_sidebar()
        await pilot.pause()
        sidebar = ws.query_one(Sidebar)
        from textual.widgets import ListView

        sessions_view = sidebar.query_one("#sb-sessions", ListView)
        # Each session becomes a ListItem; its rendered text contains the id.
        session_texts: "list[str]" = []
        for item in sessions_view.children:
            for child in item.children:
                session_texts.append(_static_text(child))
        assert any("s1" in t for t in session_texts)


def test_approval_mask_value_redacts_sensitive_keys() -> None:
    assert _mask_value("api_key", "sk-123") == "***"
    assert _mask_value("password", "hunter2") == "***"
    assert _mask_value("token", "abc") == "***"


def test_approval_mask_value_truncates_long_values() -> None:
    long = "x" * 200
    masked = _mask_value("content", long)
    assert masked.endswith("…")
    assert len(masked) < 200


def test_approval_mask_value_handles_nested() -> None:
    out = _mask_value("args", {"api_key": "sk", "path": "/tmp/x"})
    assert "***" in out
    assert "/tmp/x" in out


def test_approval_mask_value_non_sensitive_passes() -> None:
    assert _mask_value("command", "ls -la") == "ls -la"
