#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""linktools.ai.governance: the converged security + policy decision
boundary (Phase 6 op 1). ``governance.security`` (SecurityPipeline,
SecurityBaseline, AuthorizationService, redaction, audit events) and
``governance.policy`` (PolicyEngine and its rule modules) are the two
sub-packages that together decide whether/how a tool call or sensitive
operation proceeds; ManagedToolAdapter and GovernedToolInvoker are the
call sites that consult both."""
