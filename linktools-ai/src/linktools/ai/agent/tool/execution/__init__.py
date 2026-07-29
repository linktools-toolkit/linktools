"""The single governed tool execution path."""

from .binding import ToolExecutionBinding, ToolRevisionSet
from .models import ExecuteTool, ToolExecutionContext

__all__ = [
    "ExecuteTool",
    "ToolExecutionBinding",
    "ToolExecutionContext",
    "ToolExecutionService",
    "ToolRevisionSet",
]


def __getattr__(name: str):
    if name == "ToolExecutionService":
        from .service import ToolExecutionService

        return ToolExecutionService
    raise AttributeError(name)
