#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Narrow Activity binding interceptor boundary."""

from .scope import ActivityScope

class ActivityScopeInterceptor:
    """Install a supplied binding only for one Activity invocation."""

    async def intercept_activity(self, activity: object, binding: object, *args: object, **kwargs: object) -> object:
        token = ActivityScope.bind(binding)
        try:
            return await activity(*args, **kwargs)
        finally:
            ActivityScope.reset(token)


__all__ = ["ActivityScopeInterceptor"]
