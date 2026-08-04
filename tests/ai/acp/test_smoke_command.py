import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from linktools.commands.ai.smoke import Command, SmokeClient


def test_smoke_command_exposes_real_process_options() -> None:
    with pytest.raises(SystemExit) as raised:
        Command()(["--help"])
    assert raised.value.code == 0


@pytest.mark.asyncio
async def test_smoke_approval_policy_selects_semantic_option() -> None:
    import acp.schema as schema

    options = [
        schema.PermissionOption(optionId="reject", name="Reject", kind="reject_once"),
        schema.PermissionOption(optionId="allow", name="Allow", kind="allow_once"),
    ]
    client = SmokeClient([], "deny")
    response = await client.request_permission("s1", object(), options)

    assert response.outcome.option_id == "reject"
    assert client.selected_permission_kinds == ["reject_once"]


@pytest.mark.parametrize(
    ("prompt", "approval", "expected_tool_calls", "expected_stop"),
    (
        ("SMOKE_TEXT", "deny", 0, "end_turn"),
        ("SMOKE_TOOL_ALLOW", "allow", 1, "end_turn"),
        ("SMOKE_TOOL_DENY", "deny", 0, "cancelled"),
    ),
)
def test_smoke_runs_real_acp_subprocess(
    prompt: str,
    approval: str,
    expected_tool_calls: int,
    expected_stop: str,
) -> None:
    root = Path(__file__).resolve().parents[3]
    fixture = root / "tests" / "fixtures" / "acp_smoke_project"
    env = dict(os.environ)
    env["DEBUG"] = "false"
    env["PYTHONPATH"] = os.pathsep.join(
        (str(root / "linktools-ai" / "src"), str(root / "linktools" / "src"))
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "linktools",
            "ai",
            "smoke",
            "--project",
            str(fixture),
            "--prompt",
            prompt,
            "--approval",
            approval,
            "--json",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["ok"] is True
    assert result["stop_reason"] == expected_stop
    assert result["tool_call_count"] == expected_tool_calls
    if prompt == "SMOKE_TEXT":
        assert result["message_chunk_count"] == 2
    if prompt.startswith("SMOKE_TOOL"):
        assert result["permission_request_count"] == 1
