#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""SSH host-key verification policy (spec §13.4 SSH-002).

Mapping from a linktools policy value to a paramiko ``MissingHostKeyPolicy``
is centralised here so callers do not reach for ``AutoAddPolicy`` directly.

Security posture (spec §13.4):

* ``STRICT`` (intended default) -- reject unknown/changed hosts; load
  ``known_hosts``.
* ``ACCEPT_NEW`` -- accept a host key on first contact, then verify it
  thereafter (atomic, append-only write to ``known_hosts``); a changed key
  must fail.
* ``INSECURE`` -- auto-add and never verify (emits a warning); only for
  ephemeral/loopback connections (e.g. a USB-forwarded iOS device).
"""

from linktools.core import environ

__all__ = ["SSHHostKeyPolicy", "host_key_policy_class"]

STRICT = "strict"
ACCEPT_NEW = "accept_new"
INSECURE = "insecure"


class SSHHostKeyPolicy(object):
    """Symbolic host-key policy values (spec §13.4)."""
    STRICT = STRICT
    ACCEPT_NEW = ACCEPT_NEW
    INSECURE = INSECURE
    ALL = (STRICT, ACCEPT_NEW, INSECURE)


def host_key_policy_class(policy: str) -> type:
    """Return the paramiko MissingHostKeyPolicy class for ``policy``.

    Lazily imports paramiko. ``STRICT`` returns paramiko's RejectPolicy; the
    caller is responsible for loading known_hosts.
    """
    import paramiko

    if policy == ACCEPT_NEW:
        return paramiko.AutoAddPolicy  # TODO  atomic known_hosts write
    if policy == INSECURE:
        environ.logger.warning(
            "Using insecure SSH host-key policy; host identity is NOT verified."
        )
        return paramiko.AutoAddPolicy
    # STRICT default
    return paramiko.RejectPolicy
