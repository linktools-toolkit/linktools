#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The wheel-only external-adapter conformance package.

A standalone test package that proves a THIRD PARTY can install the built
``linktools-ai`` wheel + the ``test`` extra and run the storage conformance
testkit Contracts against a from-scratch adapter -- with NO source-tree
access and NO in-repo relative imports. ``adapter.py`` imports only the
installed ``linktools.ai.*`` public surface; the testkit itself (``testing``,
a sibling of this package, resolved via ``conftest.py``) is test-support code
that ships alongside the wheel rather than inside it.

Run (after ``pip install linktools-ai[test]``, with the sibling ``testing/``
package alongside this one)::

    pytest linktools-ai/conformance/
"""
