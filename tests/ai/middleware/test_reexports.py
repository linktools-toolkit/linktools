#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""linktools.ai.middleware's __init__.py had zero exports at all. This test
proves the shallow re-export resolves to the exact same object as the deep
submodule import."""


def test_middleware_reexport_identity():
    from linktools.ai.middleware import Middleware as Shallow
    from linktools.ai.middleware.base import Middleware as Deep
    assert Shallow is Deep


def test_middleware_pipeline_reexport_identity():
    from linktools.ai.middleware import MiddlewarePipeline as Shallow
    from linktools.ai.middleware.pipeline import MiddlewarePipeline as Deep
    assert Shallow is Deep
