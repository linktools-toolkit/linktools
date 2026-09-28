#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""SSH host-key verification policy.

Maps a linktools policy value to a paramiko ``MissingHostKeyPolicy`` so callers
do not reach for ``AutoAddPolicy`` directly. This helper only selects the
policy class; it does not load ``known_hosts`` or persist host keys itself.

* ``STRICT`` (intended default) -- paramiko ``RejectPolicy``; unknown and
  changed host keys are rejected.
* ``ACCEPT_NEW`` -- paramiko ``AutoAddPolicy``; accepts an unknown host key on
  first contact.
* ``INSECURE`` -- paramiko ``AutoAddPolicy`` (emits a warning); auto-adds and
  never verifies. Only for ephemeral/loopback connections (e.g. a
  USB-forwarded iOS device).
"""

from linktools.core import environ

__all__ = ["SSHHostKeyPolicy", "host_key_policy_class"]

STRICT = "strict"
ACCEPT_NEW = "accept_new"
INSECURE = "insecure"


class SSHHostKeyPolicy(object):
    """Symbolic host-key policy values."""
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
        return paramiko.AutoAddPolicy
    if policy == INSECURE:
        environ.logger.warning(
            "Using insecure SSH host-key policy; host identity is NOT verified."
        )
        return paramiko.AutoAddPolicy
    # STRICT default
    return paramiko.RejectPolicy
