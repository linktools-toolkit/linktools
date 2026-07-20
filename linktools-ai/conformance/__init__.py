#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The wheel-only external-adapter conformance package (plan §7.2 line 1852).

A standalone test package that proves a THIRD PARTY can install the built
``linktools-ai`` wheel + the ``test`` extra and run the public storage testkit
Contracts against a from-scratch adapter -- with NO source-tree access and NO
in-repo relative imports. Everything here imports either the installed
``linktools.ai.*`` public surface or a sibling module within this package.

Run (after ``pip install linktools-ai[test]``)::

    pytest linktools-ai/conformance/
"""
