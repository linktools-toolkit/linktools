"""Semantic model-call recording at the model boundary."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import asyncio
from typing import Any

from pydantic_ai.models import Model

class SemanticRecordingModel(Model):
    def __init__(self, delegate: Model, collector: Any, codec: Any) -> None:
        object.__setattr__(self, "_delegate", delegate)
        object.__setattr__(self, "_collector", collector)
        object.__setattr__(self, "_codec", codec)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    @property
    def model_name(self) -> str:
        return self._delegate.model_name

    @property
    def system(self) -> str:
        return self._delegate.system

    def _request(self, messages: Any, settings: Any, parameters: Any) -> dict[str, Any]:
        return dict(self._codec.encode_model_request(tuple(messages), settings, parameters))

    async def request(self, messages, model_settings, model_request_parameters):  # type: ignore[override]
        started = datetime.now(timezone.utc)
        request = self._request(messages, model_settings, model_request_parameters)
        try:
            response = await self._delegate.request(messages, model_settings, model_request_parameters)
        except Exception as exc:
            await self._collector.model_request_failed({"request": request, "status": "failed", "error": {"error_type": type(exc).__name__, "message": str(exc)}, "started_at": started.isoformat()})
            raise
        await self._collector.model_request_succeeded({"request": request, "response": self._codec.encode_model_response(response), "status": "completed", "started_at": started.isoformat(), "completed_at": datetime.now(timezone.utc).isoformat()})
        return response

    @asynccontextmanager
    async def request_stream(self, messages, model_settings, model_request_parameters, run_context=None):  # type: ignore[override]
        started = datetime.now(timezone.utc)
        request = self._request(messages, model_settings, model_request_parameters)
        try:
            async with self._delegate.request_stream(messages, model_settings, model_request_parameters, run_context) as response:
                yield response
            await self._collector.model_request_succeeded({"request": request, "response": self._codec.encode_model_response(response.get()), "status": "completed", "started_at": started.isoformat(), "completed_at": datetime.now(timezone.utc).isoformat()})
        except BaseException as exc:
            status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
            await self._collector.model_request_failed({"request": request, "status": status, "error": {"error_type": type(exc).__name__, "message": str(exc)}, "started_at": started.isoformat()})
            raise


__all__ = ["SemanticRecordingModel"]
