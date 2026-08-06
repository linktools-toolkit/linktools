#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Logfire/OTel configuration boundary; library code does not configure it."""


class LogfireInstrumentation:
    def __init__(self, telemetry: object) -> None:
        self._telemetry = telemetry

    def span(self, name: str) -> object:
        return self._telemetry.span(name)


__all__ = ["LogfireInstrumentation"]
