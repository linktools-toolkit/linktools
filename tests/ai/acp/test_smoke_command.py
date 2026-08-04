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
