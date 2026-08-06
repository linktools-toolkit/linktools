#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Temporal Worker composition boundary."""

from ..temporal.worker import TemporalSdkClient, TemporalSdkWorker, TemporalWorker, WorkerActivities, WorkerRegistration, build_production_worker


def register_worker(
    worker: TemporalWorker,
    registration: WorkerRegistration,
) -> None:
    registration.register(worker)


def build_worker(
    client: TemporalSdkClient,
    activities: WorkerActivities,
) -> TemporalSdkWorker:
    return build_production_worker(client, activities)


__all__ = ["build_worker", "register_worker"]
