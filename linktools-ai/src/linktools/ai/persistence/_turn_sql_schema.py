#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Logical addresses for persisted source turn coordination."""

RUNTIME_TURN_SCOPE = "runtime-pending-turn"
TURN_SESSION_MAX_BYTES = 128
TURN_REQUEST_MAX_BYTES = 64
TURN_EXECUTION_MAX_BYTES = 128



def unoccupied_turn_route(key_locator: str) -> bytes:
    """Disjoint 33-byte absence token for this unique physical row.

    Request routes are ASCII lowercase hex; occupied session routes are UTF-8.
    Neither valid alphabet can start with 0xff. The row's existing locator is
    unique within its store, so distinct unoccupied rows remain insertable.
    """
    return b"\xff" + bytes.fromhex(key_locator)
