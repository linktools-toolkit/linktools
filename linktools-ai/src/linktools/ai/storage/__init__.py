#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Storage kernel: the lowest layer of the storage stack. Holds only generic
storage-kernel machinery -- object / cache / blob / coordination / backends --
with NO dependency on any domain package (asset / artifact / run / jobs /
runtime / capability). Domains depend on this kernel's narrow Protocols; the
kernel never depends back.

The runtime composition (``Storage`` + ``FilesystemStorage`` +
``SqlAlchemyStorage`` + ``StorageFeatures`` + ``StorageUnitOfWork``) lives at
``linktools.ai.runtime.persistence``. Importing ``linktools.ai.storage`` pulls
only the storage kernel; it never pulls a domain package or a runtime
composition."""

__all__: "list[str]" = []
