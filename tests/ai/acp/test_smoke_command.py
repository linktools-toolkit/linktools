import pytest

from linktools.commands.ai.smoke import Command


def test_smoke_command_exposes_real_process_options() -> None:
    with pytest.raises(SystemExit) as raised:
        Command()(["--help"])
    assert raised.value.code == 0
