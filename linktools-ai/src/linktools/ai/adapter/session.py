#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session persistence adapter boundary."""

from typing import Protocol

from ..core import Principal
from ..runtime.services import SessionView


class SessionGateway(Protocol):
    async def get_session(self, session_id: str, *, principal: Principal) -> SessionView: ...


__all__ = ["SessionGateway"]
