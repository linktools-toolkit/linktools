        if (
            not isinstance(manifest, Mapping)
            or manifest.get("kind") != "asset-snapshot"
            or manifest.get("format_version") != 2
            or not isinstance(manifest.get("entries"), list)
        ):
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
