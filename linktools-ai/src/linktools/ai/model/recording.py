"""Semantic model-call recording placed after security and before providers."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from pydantic_ai.models import Model

from ..execution.trace import SemanticTraceCollector


def _json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return value
    dump = getattr(value, "model_dump", None)
    if dump is not None:
        return dump(mode="json")
    return repr(value)


def _request_payload(messages: Any, settings: Any, parameters: Any) -> dict[str, Any]:
    return {
        "messages": _json(messages),
        "settings": _json(settings),
        "parameters": _json(parameters),
    }


class SemanticRecordingModel(Model):
    """Record each completed request while delegating model semantics."""

    def __init__(self, delegate: Model, collector: SemanticTraceCollector) -> None:
        object.__setattr__(self, "_delegate", delegate)
        object.__setattr__(self, "_collector", collector)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    @property
    def model_name(self) -> str:
        return self._delegate.model_name

    @property
    def system(self) -> str:
        return self._delegate.system

    async def request(self, messages, model_settings, model_request_parameters):  # type: ignore[override]
        started = datetime.now(timezone.utc).isoformat()
        request = _request_payload(messages, model_settings, model_request_parameters)
        try:
            response = await self._delegate.request(messages, model_settings, model_request_parameters)
        except Exception as exc:
            await self._collector.model_request_failed({"request": request, "status": "error", "error": {"type": type(exc).__name__, "message": str(exc)}, "started_at": started})
            raise
        await self._collector.model_request_succeeded({"request": request, "response": _json(response), "status": "completed", "started_at": started, "completed_at": datetime.now(timezone.utc).isoformat()})
        return response

    @asynccontextmanager
    async def request_stream(self, messages, model_settings, model_request_parameters, run_context=None):  # type: ignore[override]
        started = datetime.now(timezone.utc).isoformat()
        request = _request_payload(messages, model_settings, model_request_parameters)
        try:
            async with self._delegate.request_stream(messages, model_settings, model_request_parameters, run_context) as response:
                yield response
            await self._collector.model_request_succeeded({"request": request, "response": _json(response.get()), "status": "completed", "started_at": started, "completed_at": datetime.now(timezone.utc).isoformat()})
        except Exception as exc:
            await self._collector.model_request_failed({"request": request, "status": "error", "error": {"type": type(exc).__name__, "message": str(exc)}, "started_at": started})
            raise


__all__ = ["SemanticRecordingModel"]
