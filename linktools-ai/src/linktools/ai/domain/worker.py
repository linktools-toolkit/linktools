#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pinned Worker Route metadata."""

from pydantic import BaseModel, ConfigDict


PAYLOAD_DECODER_VERSION = 1


class WorkerRoute(BaseModel):
    """Bundle-compatible Worker deployment record."""

    model_config = ConfigDict(frozen=True)

    bundle_id: str
    task_queue: str
    worker_deployment: str
    current_build: str
    payload_decoder_versions: "frozenset[int]"
    healthy: bool = True

    def is_compatible(self, decoder_version: int) -> bool:
        """Return whether this route can decode a workflow payload."""
        return self.healthy and decoder_version in self.payload_decoder_versions
