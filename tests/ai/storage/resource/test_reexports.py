#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""linktools.ai.storage.resource's __init__.py was an empty shell; these
symbols were only reachable via the deep submodule paths. This test proves
the shallow re-export resolves to the exact same object, not a copy."""


def test_resource_store_reexport_identity():
    from linktools.ai.storage.resource import ResourceStore as Shallow
    from linktools.ai.storage.resource.store import ResourceStore as Deep
    assert Shallow is Deep


def test_resource_path_reexport_identity():
    from linktools.ai.storage.resource import ResourcePath as Shallow
    from linktools.ai.storage.resource.path import ResourcePath as Deep
    assert Shallow is Deep


def test_resource_models_reexport_identity():
    from linktools.ai.storage.resource import Found as FoundShallow, Masked as MaskedShallow, Missing as MissingShallow, WriteOptions as WriteOptionsShallow
    from linktools.ai.storage.resource.models import Found as FoundDeep, Masked as MaskedDeep, Missing as MissingDeep, WriteOptions as WriteOptionsDeep
    assert FoundShallow is FoundDeep
    assert MaskedShallow is MaskedDeep
    assert MissingShallow is MissingDeep
    assert WriteOptionsShallow is WriteOptionsDeep
