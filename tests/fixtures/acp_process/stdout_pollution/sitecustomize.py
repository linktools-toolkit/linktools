#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Emit a framing violation before the ACP child starts."""

import sys

sys.stdout.write("unexpected child stdout\n")
sys.stdout.flush()
