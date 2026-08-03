#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The right context panel.

Shows the active agent's resolved features (tools/skills/mcp/extensions), the
doctor verdict summary, and the current model. A persistent panel (unlike the
old full-screen CatalogScreen/DoctorScreen overlays) that refreshes via
``render_*`` without taking over the conversation."""

from typing import TYPE_CHECKING, Any

from rich.markup import escape
from textual.containers import Vertical
from textual.widgets import Label, Static

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from ..client import DoctorReport


class ContextPanel(Vertical):
    """Active agent + doctor + model snapshot.

    ``render_inspection`` takes whatever ``RuntimeClient.inspect`` returns and
    formats the tool/skill/mcp/extension counts without assuming a fixed
    schema -- the runtime may omit sections."""

    DEFAULT_CSS = """
    ContextPanel {
        width: 32;
        min-width: 22;
        max-width: 48;
        layout: vertical;
        background: $panel;
        border-left: solid $primary;
        overflow-y: auto;
    }
    ContextPanel .cp-section { height: auto; margin: 0 0 1 0; padding: 0 1; }
    ContextPanel .cp-title { color: $text-muted; text-style: bold; margin-bottom: 0; }
    ContextPanel .cp-value { color: $text; padding: 0 1; }
    ContextPanel .cp-empty { color: $text-disabled; padding: 0 1; }
    """

    def compose(self) -> "ComposeResult":
        with Vertical(classes="cp-section"):
            yield Label("Agent", classes="cp-title")
            yield Static("(no agent)", id="cp-agent", classes="cp-empty")
        with Vertical(classes="cp-section"):
            yield Label("Doctor", classes="cp-title")
            yield Static("(not run)", id="cp-doctor", classes="cp-empty")
        with Vertical(classes="cp-section"):
            yield Label("Model", classes="cp-title")
            yield Static("(unset)", id="cp-model", classes="cp-empty")

    def render_inspection(self, inspection: Any, *, agent_id: "str | None") -> None:
        widget = self.query_one("#cp-agent", Static)
        widget.remove_class("cp-empty")
        if inspection is None:
            widget.update(escape(agent_id or "(no agent)"))
            return
        sections: "list[str]" = []
        for attr, label in (
            ("tools", "tools"),
            ("skills", "skills"),
            ("mcp_servers", "mcp"),
            ("extensions", "extensions"),
            ("features", "features"),
        ):
            value = getattr(inspection, attr, None)
            if value is None and isinstance(inspection, dict):
                value = inspection.get(attr)
            if not value:
                continue
            sections.append(f"{label}: {len(value)}")
        head = escape(agent_id or getattr(inspection, "id", "") or "(agent)")
        widget.update(
            head + (f"\n{' · '.join(escape(s) for s in sections)}" if sections else "")
        )

    def render_doctor(self, report: "DoctorReport | None") -> None:
        widget = self.query_one("#cp-doctor", Static)
        if report is None:
            widget.update("(not run)")
            widget.add_class("cp-empty")
            return
        widget.remove_class("cp-empty")
        total = len(report.checks)
        failed = len(report.failed)
        widget.update(
            escape(
                f"{total - failed}/{total} ok"
                + (f" ({failed} failed)" if failed else "")
            )
        )

    def render_model(self, model: "str | None", *, config: "str | None" = None) -> None:
        widget = self.query_one("#cp-model", Static)
        if not model and not config:
            widget.update("(unset)")
            widget.add_class("cp-empty")
            return
        widget.remove_class("cp-empty")
        widget.update(escape(" · ".join(p for p in (model, config) if p)))
