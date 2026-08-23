"""Raw Asset file storage."""

from ..errors import (
    AssetConflictError,
    AssetError,
    AssetNotFoundError,
    AssetParseError,
    InvalidAssetError,
)
from ._backend import InMemoryAssetBackend
from ._config import StrictConfigReader, resolved_name
from ._directory import (
    AssetPathAdapter,
    DirectoryAssetBackend,
    PrefixAssetPathAdapter,
    directory_root,
)
from ._domain import (
    AssetBackend,
    AssetInfo,
    AssetKey,
    AssetRoot,
    WritableAssetBackend,
)
from ._filesystem import FilesystemAssetBackend, filesystem_root
from ._logical import (
    AssetCodec,
    AssetDiscoveryStatus,
    AssetEntry,
    AssetRef,
    AssetResource,
    AssetRetargeter,
    AssetTypeBinding,
    AssetTypeRegistry,
    AssetTypeRegistrySnapshot,
    AssetValueAdapter,
    AssetVariantBinding,
    DirectoryLayout,
    ResolvedAsset,
    SingleFileLayout,
)
from ._object import AssetObjectKeyFactory
from ._repository import AssetRepository, AssetScope
from ._sql import (
    SqlAssetBackend,
    build_asset_sql_metadata,
)
from ._store import AssetCacheAdapter, AssetStore

__all__ = [
    "AssetBackend",
    "AssetCacheAdapter",
    "AssetCodec",
    "AssetConflictError",
    "AssetDiscoveryStatus",
    "AssetEntry",
    "AssetError",
    "AssetInfo",
    "AssetKey",
    "AssetNotFoundError",
    "AssetObjectKeyFactory",
    "AssetParseError",
    "AssetPathAdapter",
    "AssetRef",
    "AssetRepository",
    "AssetResource",
    "AssetRetargeter",
    "AssetRoot",
    "AssetScope",
    "AssetStore",
    "AssetTypeBinding",
    "AssetTypeRegistry",
    "AssetTypeRegistrySnapshot",
    "AssetValueAdapter",
    "AssetVariantBinding",
    "DirectoryAssetBackend",
    "DirectoryLayout",
    "FilesystemAssetBackend",
    "InMemoryAssetBackend",
    "InvalidAssetError",
    "PrefixAssetPathAdapter",
    "ResolvedAsset",
    "SingleFileLayout",
    "SqlAssetBackend",
    "StrictConfigReader",
    "WritableAssetBackend",
    "build_asset_sql_metadata",
    "directory_root",
    "filesystem_root",
    "resolved_name",
]
