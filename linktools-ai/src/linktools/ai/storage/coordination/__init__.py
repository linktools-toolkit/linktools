#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lease coordination: the LeaseCoordinator Protocol lives in
:mod:`linktools.ai.storage.protocols`; this package holds the in-repo reference
implementation(s). The reference impl is imported lazily from
``.process_local`` (not re-exported here) to avoid pulling ``storage.protocols``
during package init. A downstream deployment injects its own distributed
coordinator (Redis/etcd/a shared DB lease table) implementing the same Protocol.
"""
