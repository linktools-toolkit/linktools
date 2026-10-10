# Container operation lifecycle

The public context is `OperationContext`. A container uses the existing
`on_starting` and `on_check` callbacks; it does not implement a separate generated
configuration lifecycle.

## Ordering and responsibilities

An `up` operation resolves the selected groups, required providers and declared
running consumers. Container requirements select what participates; only Compose
service dependency edges determine startup order. A real Compose dependency cycle
is an error. There is no implicit bootstrap configuration.

The operation then runs:

1. `on_starting`, container `BEFORE_START`, manager `BEFORE_START`: prepare build
   inputs, persistent secrets and immutable configuration inputs for the selected
   owners. Do not change a running service or another owner's lifecycle.
2. Image preparation, using the resolved model and the existing image preparer.
3. `on_check` and `CHECK`: validate the exact prepared inputs with the target
   images. No service has been stopped or replaced at this point.
4. Framework-managed service application in Compose dependency order, followed by
   readiness confirmation and per-service applied-model recording.
5. `on_started` and `AFTER_START`: post-application notifications. A failure here
   reports that services were applied; it does not trigger deployment rollback.

`restart` performs its explicit stop only after step 3. Required providers that
were not explicitly selected are not included in the stop set. `down` only uses
the stop callbacks and performs no start preparation, build or native check.
Status reads metadata and actual state without invoking preparation callbacks.

Existing callbacks are retained for loading, stopping and removal. The former
`on_prepare` side-effectful loading hook is removed. The existing seven
`HookPhase` values are unchanged.

## Preparing files

Use `context.write_files(self, files, mode=0o600, group=None)` once during
`on_starting`. `files` maps relative names to UTF-8 text. The call returns an
immutable host directory and does not publish it to running services.

Compose templates reference the corresponding logical sources under
`APP_PATH / "generated/current"`. The framework resolves selected file mounts to
immutable prepared inputs before image checks or service application. A native
check reads the same mounts, or uses `context.file_path(self, name)` for a local
format check. Check callbacks do not overwrite prepared inputs.

Prefer an individual file bind for a service that consumes only one file. When
another file changes, unchanged single-file consumers retain their prior mount
source and are not forced to recreate. A directory bind consumes the complete
prepared tree. Missing inputs or modified immutable trees cause an explicit
failure before application.

Persistent secrets, databases, user assets and ACME account state are not ordinary
prepared configuration. They are not replaced by this file mechanism.

## Applying and recovering

Changed file mounts or Compose definitions recreate the affected service.
Unchanged definitions are not force-recreated. Docker Compose can still replace a
service when its locally selected image changes. Recreating a container does not
by itself rebuild its image or request a new certificate.

Readiness follows effective Docker health checks and Compose dependency
conditions. Successful finite jobs are recorded as applied, not running.
Dependency/job readiness has no newly imposed fixed task deadline. A failure
blocks subsequent dependents.

The per-service resolved applied model is the sole recovery authority. Recovery
pins the pre-operation image ID and restores the failed service, any explicitly
stopped services not yet reapplied, and related consumers sharing a changed file
input. Newly started related consumers without a prior instance are stopped.
Independent successful siblings are retained. Initial migration may read old saved Compose files, but
new operations do not write a second container-level applied snapshot.

Running new containers bind immutable inputs, not `generated/current`. The latter
is a convenience reference, updated after successful application and not changed
while an unselected legacy service still dereferences it. Applied snapshots, not
this convenience link, define the actual per-service input.

A failed recovery reports both the operation and recovery errors. The framework
never claims to undo database or external-system changes.

## Built-in services

LLDAP and Authelia prepare configuration and persistent secrets in `on_starting`.
Authelia validates with its target image. Its admin service mounts only the base
configuration; ACL-only changes do not change that service's file mount.

Flare prepares navigation files with its configured group access. Initial
migration copies other legacy application assets to `runtime-app`, leaving the
legacy directory available to an old Compose model. Application uses normal
file mounts and service recreation, not migration callbacks around Compose.

Nginx prepares a complete configuration, checks it with the target image and
starts it normally. Deployment no longer reloads a configuration generation.
Requests whose authentication or WAF upstream is unavailable must still fail
closed. Custom templates requiring startup-time upstream resolution must provide
real Compose dependencies or reject validation; the framework does not substitute
an alternative configuration to hide a cycle.

## ACME

Initial certificates are issued during a single-stage image build. DNS
credentials are written to `APP_PATH/nginx/acme-secrets/dns.env` with restricted
host permissions, used as a BuildKit secret and mounted read-only at runtime.
Only the ACME subprocess loads them into its environment. Known DNS provider
assignments are removed from saved ACME configuration before it is copied into
image state, including credentials from a previously selected provider.

The certificate revision identifies build code and TLS inputs, excluding DNS
credential values. Each revision has a separate TLS storage slot under
`/etc/certs/<revision>`. The image entrypoint installs preissued material into its
own slot; ordinary startup does not contact a CA. The restored image/configuration
therefore references its original slot without Python code changing a TLS pointer.

The existing certificate script owns all TLS pointer changes and renewal reloads,
under the shared certificate lock. Renewal verifies the served certificate and
restores the previous served certificate on failure. Host ACME-account archival
uses the same lock on supported Linux bind mounts. Certificate and account keys
remain sensitive contents of the image; do not distribute it to untrusted users.
