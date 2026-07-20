#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""linktools.ai.storage.sqlalchemy already exports SqlAlchemyStorage; this
test proves SqlAlchemyAssetBackend is now re-exported the same way,
resolving to the exact same object as the deep submodule import."""


def test_sqlalchemy_resource_backend_reexport_identity():
    from linktools.ai.storage.sqlalchemy import SqlAlchemyAssetBackend as Shallow
    from linktools.ai.storage.sqlalchemy.asset import SqlAlchemyAssetBackend as Deep
    assert Shallow is Deep


def test_existing_sqlalchemy_storage_export_still_works():
    """Regression guard: this task must not remove or break the existing export."""
    from linktools.ai.storage.sqlalchemy import SqlAlchemyStorage
    assert SqlAlchemyStorage is not None
