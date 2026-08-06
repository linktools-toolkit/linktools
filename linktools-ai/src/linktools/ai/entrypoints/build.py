#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build composition root."""


def build_artifacts(builder: object, request: object) -> object:
    return builder.build(request)


__all__ = ["build_artifacts"]
