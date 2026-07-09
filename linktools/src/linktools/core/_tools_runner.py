"""ToolRunner: execute a resolved managed tool via the runtime Process."""

from typing import TYPE_CHECKING

from ..runtime import popen

if TYPE_CHECKING:
    from typing import Any, Sequence
    from ..types import TimeoutType

__all__ = ["ResolvedTool", "ToolRunner"]


class ResolvedTool:
    """A tool resolved to a concrete executable + environment."""

    def __init__(self, executable: str, env: "dict[str, str] | None" = None,
                 source: str = "managed", version: "str | None" = None) -> None:
        self.executable = executable
        self.env = dict(env or {})
        self.source = source
        self.version = version


class ToolRunner:
    """Runs resolved tools through the runtime Process."""

    def __init__(self, environ: "Any") -> None:
        self._environ = environ

    def popen(self, resolved: "ResolvedTool", args: "Sequence[str]" = (), *,
              include_tools: bool = True, env_overrides: "dict[str, str] | None" = None,
              **kwargs: "Any") -> "Any":
        """Spawn the resolved tool; return the runtime Process (do not wait)."""
        env = self._environ.subprocess_env(
            include_tools=include_tools, overrides=env_overrides)
        env.update(resolved.env)
        command = [resolved.executable] + [str(a) for a in args]
        return popen(*command, env=env, **kwargs)

    def run(self, resolved: "ResolvedTool", args: "Sequence[str]" = (), *,
            check: bool = True, timeout: "TimeoutType" = None, **kwargs: "Any") -> int:
        """Run the resolved tool to completion; return its exit code."""
        process = self.popen(resolved, args, **kwargs)
        if check:
            return process.check_call(timeout=timeout)
        return process.call(timeout=timeout)
