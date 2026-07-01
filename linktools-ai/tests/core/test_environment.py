import logging
from pathlib import Path

from linktools.ai.core.environment import AgentEnvironment
from linktools.ai.support.hooks import HookRegistry


class _MinimalEnv:
    def __init__(self) -> None:
        self.hooks = HookRegistry()
        self.env = "stg"
        self.config_root = Path("/tmp/config")
        self.workspace_root = Path("/tmp/workspace")

    def get_logger(self, name: str) -> logging.Logger:
        return logging.getLogger(name)

    def trace_root(self, trace_id: str) -> Path:
        return self.workspace_root / trace_id


def test_minimal_env_satisfies_protocol():
    assert isinstance(_MinimalEnv(), AgentEnvironment)
