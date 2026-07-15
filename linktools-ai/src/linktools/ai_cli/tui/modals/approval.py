#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The approval modal.

When a run pauses for approval the chat screen pushes this modal. It shows the
run/approval ids, tool name, reason and masked arguments, and offers Approve /
Reject / Later. The decision is returned to the screen via ``dismiss(...)``;
the screen owns the side effects (approve+resume, reject+cancel, or leave the
run waiting)."""

from typing import TYPE_CHECKING, Any, Mapping

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

if TYPE_CHECKING:
    from linktools.ai_cli.client import RuntimeClient

_APPROVE = "approve"
_REJECT = "reject"
_LATER = "later"

# Heuristic argument masking for the approval modal. Redacts values whose key
# looks sensitive and truncates long ones. The TUI cannot know every secret
# field, so this is best-effort defense -- a real masking policy belongs in the
# backend approval layer.
_SENSITIVE_HINTS = (
    "token",
    "secret",
    "password",
    "passwd",
    "auth",
    "credential",
    "api_key",
    "apikey",
)
_MAX_ARG_VALUE = 80


def _looks_sensitive(key: str) -> bool:
    key_lower = key.lower()
    return any(hint in key_lower for hint in _SENSITIVE_HINTS)


def _mask_value(key: str, value: Any) -> str:
    if _looks_sensitive(key):
        return "***"
    if isinstance(value, dict):
        return (
            "{"
            + ", ".join(f"{k}: {_mask_value(str(k), v)}" for k, v in value.items())
            + "}"
        )
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_mask_value("", v) for v in value) + "]"
    text = value if isinstance(value, str) else repr(value)
    if len(text) > _MAX_ARG_VALUE:
        text = text[:_MAX_ARG_VALUE] + "…"
    return text


def _render_arguments(arguments: "Mapping[str, Any]") -> str:
    if not arguments:
        return "  [dim](none)[/dim]"
    return "\n".join(
        f"  {escape(str(k))}: {escape(_mask_value(str(k), v))}"
        for k, v in arguments.items()
    )


class ApprovalModal(ModalScreen):
    """A pause-for-approval dialog. Dismisses with ``approve``/``reject``/``later``."""

    BINDINGS = [
        Binding("a", "approve", "Approve"),
        Binding("r", "reject", "Reject"),
        Binding("l,escape", "later", "Later"),
    ]

    def __init__(self, *, client: "RuntimeClient", event: "Mapping[str, Any]") -> None:
        super().__init__()
        self.client = client
        self.event = event

    def compose(self) -> ComposeResult:
        run_id = self.event.get("run_id")
        approval_id = self.event.get("approval_id")
        yield Vertical(
            Static("[b]Approval required[/b]"),
            Static(f"run id: {escape(str(run_id))}"),
            Static(f"approval id: {escape(str(approval_id))}"),
            Static("[dim]loading tool detail…[/dim]", id="approval-detail"),
            Static(""),
            Button("Approve once [a]", id=_APPROVE, variant="success"),
            Button("Reject [r]", id=_REJECT, variant="error"),
            Button("Later [l]", id=_LATER),
            id="approval-modal",
        )

    def on_mount(self) -> None:
        self.run_worker(self._load_detail())

    async def _load_detail(self) -> None:
        # Read the tool/arguments/reason through the client and fold them into
        # the detail line; a missing request degrades to the ids.
        approval_id = self.event.get("approval_id")
        try:
            request = (
                await self.client.get_approval(approval_id) if approval_id else None
            )
        except Exception:
            # The buttons still work without the detail; don't leave the user
            # staring at "loading…" forever.
            self.query_one("#approval-detail", Static).update(
                "[red]failed to load tool detail[/red]"
            )
            return
        lines = []
        if request is not None:
            tool = escape(str(getattr(request, "tool_name", "?")))
            lines.append(f"tool: {tool}")
            reason = getattr(request, "reason", None)
            if reason:
                lines.append(f"reason: {escape(str(reason))}")
            lines.append("arguments:")
            lines.append(_render_arguments(getattr(request, "arguments", {}) or {}))
        else:
            lines.append("tool: ?")
        self.query_one("#approval-detail", Static).update("\n".join(lines))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id)

    def action_approve(self) -> None:
        self.dismiss(_APPROVE)

    def action_reject(self) -> None:
        self.dismiss(_REJECT)

    def action_later(self) -> None:
        self.dismiss(_LATER)
