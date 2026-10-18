#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Host-owned, single-event-loop admission limits for logical model requests.

Run this file with Pydantic AI installed for an offline demonstration. In a
LinkTools application, register the same instance with a RuntimeFeatures:

    limits = GlobalRateLimit(
        quota_key="provider-project-a", rpm=60, tpm=60_000,
        token_reservation=4_000,
    )
    group.capability(limits, id="host.rate-limit", scope="runtime")

Pass that group in Runtime.open(..., features=group). The quota key is an
explicit host label for the project/account bucket, not a model or credential
lookup. Share this instance to share its counters; two instances with the same
key do not coordinate. Multiple processes need an application-owned service.

RPM counts admissions in the preceding minute. TPM holds a reservation
through the request and for one minute after it finishes, including failures
and cancellation. A successful response can increase that charge, never refund
it. This intentionally conservative workload policy is not a provider's exact
quota formula: OpenAI can charge from max_tokens/request estimates, Claude has
separate input/output limits, and Gemini TPM counts input. Choose a reservation
that covers the final prompt, tool schema, output allowance and hook mutations.
An underestimated reservation can overshoot before observed usage adds debt.

Native hooks cover logical agent requests, not every SDK retry, fallback or
continuation call, nor separate compaction calls. A returned response may have
been replaced by another hook; missing/partial usage is not evidence of zero
provider usage. Exact provider-attempt accounting requires lower-level support.

token_reservation also accepts a synchronous function of (ctx, request_context).
It can use ctx.deps and the native model, messages, settings, and tool parameters
to choose a positive integer reservation no larger than tpm. It runs once before
queuing; downstream hooks may still change the eventual request, so the host
must bound those changes when choosing its estimate.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, WrapModelRequestHandler
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage


@dataclass
class _TokenCharge:
    tokens: int
    expires_at: float | None = None


class GlobalRateLimit(AbstractCapability[object]):
    """One application quota bucket shared across runs in one event loop.

    State is deliberately local to this example, with no LinkTools backend or
    ModelRegistry contract. Keep the instance and its configuration unchanged
    while runs are active; never copy it per run or reuse it across event loops.
    """

    def __init__(
        self,
        *,
        quota_key: str,
        rpm: int,
        tpm: int,
        token_reservation: int | Callable[[RunContext[object], ModelRequestContext], int],
    ) -> None:
        super().__init__()
        if not isinstance(quota_key, str) or not quota_key.strip():
            raise ValueError("quota_key must be a non-empty host quota label")
        for name, value in (("rpm", rpm), ("tpm", tpm)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.quota_key = quota_key
        self.rpm = rpm
        self.tpm = tpm
        self.token_reservation = token_reservation
        if not callable(token_reservation):
            self._validate_reservation(token_reservation)
        self._lock = asyncio.Lock()
        self._changed = asyncio.Event()
        self._admissions: deque[float] = deque()
        self._charges: list[_TokenCharge] = []

    async def wrap_model_request(
        self,
        ctx: RunContext[object],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        reserved_tokens = (
            self.token_reservation(ctx, request_context)
            if callable(self.token_reservation) else self.token_reservation
        )
        self._validate_reservation(reserved_tokens)
        charge = await self._acquire(reserved_tokens)
        observed_tokens = 0
        try:
            response = await handler(request_context)
            observed_tokens = response.usage.total_tokens
            return response
        finally:
            self._finish(charge, observed_tokens)

    def _validate_reservation(self, tokens: int) -> None:
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            raise ValueError("token_reservation must produce a positive integer")
        if tokens > self.tpm:
            raise ValueError("token_reservation must fit within tpm")

    async def _acquire(self, reserved_tokens: int) -> _TokenCharge:
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._admissions and self._admissions[0] + 60 <= now:
                    self._admissions.popleft()
                self._charges = [
                    charge for charge in self._charges
                    if charge.expires_at is None or charge.expires_at > now
                ]
                charged_tokens = sum(charge.tokens for charge in self._charges)
                if (
                    len(self._admissions) < self.rpm
                    and charged_tokens + reserved_tokens <= self.tpm
                ):
                    charge = _TokenCharge(reserved_tokens)
                    self._admissions.append(now)
                    self._charges.append(charge)
                    return charge

                expirations = [
                    charge.expires_at for charge in self._charges
                    if charge.expires_at is not None
                ]
                if self._admissions:
                    expirations.append(self._admissions[0] + 60)
                delay = min(expirations) - now if expirations else None
                changed = self._changed

            # No charge exists while queued. Cancellation therefore cannot
            # refund or release another request's reservation.
            if delay is None:
                await changed.wait()
            else:
                try:
                    await asyncio.wait_for(changed.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass

    def _finish(self, charge: _TokenCharge, observed_tokens: int) -> None:
        # All state lives in one event loop. The locked admission section never
        # awaits, so this synchronous settlement cannot race it and cannot be
        # interrupted by cancellation during asynchronous cleanup.
        charge.tokens = max(charge.tokens, observed_tokens)
        charge.expires_at = time.monotonic() + 60
        changed = self._changed
        self._changed = asyncio.Event()
        changed.set()


async def main() -> None:
    """Run two agents against one local quota bucket without provider access."""
    limits = GlobalRateLimit(
        quota_key="offline-demo", rpm=10, tpm=100, token_reservation=20,
    )

    async def answer(
        messages: list[ModelMessage], info: AgentInfo,
    ) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart("Shared quota admitted this request")],
            usage=RequestUsage(input_tokens=8, output_tokens=5),
        )

    first = Agent(FunctionModel(answer), capabilities=[limits])
    second = Agent(FunctionModel(answer), capabilities=[limits])
    results = await asyncio.gather(first.run("First"), second.run("Second"))
    for result in results:
        print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
