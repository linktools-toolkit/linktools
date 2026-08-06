#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Manifest-driven capability/profile preflight."""

from dataclasses import dataclass

from ...domain.execution import ExecutionProfile
from ...foundation.errors import ErrorCode, LinktoolsAIError


@dataclass(frozen=True, slots=True)
class CapabilityPolicy:
    temporal_mode_by_feature: "dict[str, str]"

    def validate(self, profile: ExecutionProfile, features: "tuple[str, ...]") -> None:
        if profile is ExecutionProfile.LOCAL_CODING:
            return
        for feature in features:
            if self.temporal_mode_by_feature.get(feature, "unverified") in {"local_only", "unverified"}:
                raise LinktoolsAIError(ErrorCode.CAPABILITY_DISABLED_FOR_PROFILE, feature)


__all__ = ["CapabilityPolicy"]
