#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""linktools.ai.asset's __init__.py was an empty shell; these
symbols were only reachable via the deep submodule paths. This test proves
the shallow re-export resolves to the exact same object, not a copy."""


def test_asset_store_reexport_identity():
    from linktools.ai.asset import AssetStore as Shallow
    from linktools.ai.asset.store import AssetStore as Deep
    assert Shallow is Deep


def test_resource_path_reexport_identity():
    from linktools.ai.asset import AssetPath as Shallow
    from linktools.ai.asset.path import AssetPath as Deep
    assert Shallow is Deep


def test_resource_models_reexport_identity():
    from linktools.ai.asset import Found as FoundShallow, Masked as MaskedShallow, Missing as MissingShallow, WriteOptions as WriteOptionsShallow
    from linktools.ai.asset.models import Found as FoundDeep, Masked as MaskedDeep, Missing as MissingDeep, WriteOptions as WriteOptionsDeep
    assert FoundShallow is FoundDeep
    assert MaskedShallow is MaskedDeep
    assert MissingShallow is MissingDeep
    assert WriteOptionsShallow is WriteOptionsDeep
