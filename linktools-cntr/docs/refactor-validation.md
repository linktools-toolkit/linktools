# Lifecycle convergence validation

Target source baseline: `824c9e3da2fe7ecd79ed28389464e61d79a39b1c`.

The implementation removes the container-generated configuration protocol and
uses selected preparation, target-image checks and framework-controlled
recreation. See `lifecycle.md` for operation and recovery semantics.

## Required repository gate

Run `python manage.py check linktools-cntr` on the fully applied code. Old tests
that assert generation labels, bootstrap fallback or the removed callbacks must
be migrated to equivalent file-input/recreation behavior, not merely skipped.

## Required native acceptance

Exercise Docker Compose with Nginx, Authelia, LLDAP, SafeLine and Flare: cold start,
configuration updates, unchanged input, ACL-only changes, failure before restart
stops, failed service replacement, partial restart recovery and first upgrade
from the saved old model. Exercise actual image build, CA issuance and renewal,
including credential rotation, shared-lock contention and a failed TLS reload.

Local isolated Python, shell/OpenSSL or host-Nginx tests are not substitutes for
these repository and container acceptance gates. No successful result from an
older commit establishes acceptance for this implementation.
