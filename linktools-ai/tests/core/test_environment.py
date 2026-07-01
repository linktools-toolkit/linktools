import logging
from pathlib import Path

from linktools.ai.core.environment import AgentEnvironment
from linktools.ai.support.hooks import HookRegistry


class _MinimalEnv:
    def __init__(self) -> None:
        self.hooks = HookRegistry()
        self.env = "stg"
        self.config_root = Path("/tmp/config")

    def get_logger(self, name: str) -> logging.Logger:
        return logging.getLogger(name)


def test_minimal_env_satisfies_protocol():
    assert isinstance(_MinimalEnv(), AgentEnvironment)
