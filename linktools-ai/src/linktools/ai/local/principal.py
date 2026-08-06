#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit trusted principal for the local profile."""

from ..core import ExecutionProfile, Principal


def trusted_local_principal(principal_id: str = "local") -> Principal:
    return Principal(principal_id, "local", "LOCAL_TRUSTED")


def require_local_profile(profile: ExecutionProfile) -> None:
    if profile is not ExecutionProfile.LOCAL_CODING:
        raise ValueError("trusted local principal cannot be used by a production profile")


__all__ = ["require_local_profile", "trusted_local_principal"]
