# Container operation lifecycle

The public context is `OperationContext`. A container uses the existing
`on_starting` and `on_check` callbacks; it does not implement a separate generated
configuration lifecycle.

## Operation context fields

- `actions` contains operation labels such as `up`, `restart`, and `pull`, not process arguments
- `project_containers` is the complete project scope; `target_containers` and `target_services` are the resolved operation targets
- `is_full_project` identifies an unscoped project operation. Explicitly naming every container does not set it
- `initial_runtime_state` is the pre-operation Docker inspection. Its derived `existing_services`, `running_services`, `image_ids` and `running_images` views replace separate context copies; stopped replicas never supply a running image ID
- `prepared_dirs` maps container names to immutable prepared directory roots; use `write_files` and `file_path` to create and read their contents
- `previous_compose_contents` maps captured Compose source paths to their previous YAML text for recovery
- `refresh_services` remains the explicit refresh selection; `metadata` remains hook extension data

The former fields `commands`, `containers`, `is_full_containers`, `runtime_state`,
`initial_services`, `native_running_images`, `prepared_files`, and `saved_compose`
map to the corresponding names above without aliases. These are in-memory
operation fields; persisted Compose/applied-model formats are unchanged.

## Data ownership

`ComposeOperations` owns the generate → validate → apply → recover flow. Its
immutable `ComposeSelection` fixes the action scope; `OperationContext` exposes
that scope and the working inputs to callbacks, without granting hooks authority
to select additional actions.

- `context.compose_model` is the native Compose model bound to prepared files and
  selected image revisions. The runner constructs commands from that model
- `context.initial_runtime_state` owns observed services, image IDs and exact
  namespace bindings. Derived views do not introduce another state cache
- `context.service_models` (`AppliedServiceModels`) owns immutable current and
  previous per-service YAML snapshots. `previous_model(service)` returns an
  independent decoded copy; recording and restoring retain the disk format
- `context.prepared_dirs` owns candidate immutable file trees. Publication changes
  their public references only after application; retained snapshots protect
  historical running and rollback inputs from pruning

Image preparation has one orchestration entry point. The image preparer computes
build/pull requirements; the runner executes commands; `ComposeOperations`
orders them and owns warnings, checks and failure handling. Recovery separately
follows dependent actions, required stopped namespace providers and shared file
inputs because those relationships require different restore actions.

## Ordering and responsibilities

An `up` operation resolves the selected groups, required providers and declared
running consumers, including attached navigation links. The existing artifact
index retains the source-container names of generated inputs so removing a
producer's final declaration can still update its previously running consumer.
Container requirements select what participates; only Compose
service dependency edges determine startup order. A real Compose dependency cycle
is an error. There is no implicit bootstrap configuration.

Compose resolves profiles before startup hooks. Full-project operations apply only
active services; explicit service selection enables its profiles through native
Compose. Disabled definitions remain available for orphan detection, while their
running containers and existing generated inputs stay unchanged unless a real
dependency action requires rebinding. A plan that cannot resolve native profiles
reports the unresolved selection rather than guessing startup commands.

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
Partial `up` warns before changing a service outside the explicitly requested
containers, identifying the action and dependency/configuration reason. Unchanged
services produce no such warning; recovery reports additional actions when needed.

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
Live namespace consumers are rebound after a retained provider changes, even if
an independent service later fails. Restoring a live consumer may temporarily
start its previously stopped namespace provider with the old model and image;
the provider is stopped again after recovery. Interrupting deployment runs
recovery before propagating the interrupt; interrupting recovery itself stops
that attempt immediately and may leave it incomplete.

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

Only this revision-based layout is managed automatically. Before upgrading an old
deployment, manually back up its container-local `/etc/certs` and `/root/.acme.sh`
contents. The tool does not copy old writable layers, import flat `certs/live`
directories, or add migration mounts to rollback models. Existing files and
backups are left untouched; restoring old-layout data is a manual operation.
An ACME account supplied in the host `nginx/acme` directory remains a build input
when no current revision account exists. Changing the certificate script requires
one normal image rebuild and build-time issuance; ordinary `up` adds no image pull
or runtime issuance.

The existing certificate script owns all TLS pointer changes and renewal reloads,
under the shared certificate lock. Renewal verifies the served certificate and
restores the previous served certificate on failure. Host ACME-account archival
uses the same lock on supported Linux bind mounts. Certificate and account keys
remain sensitive contents of the image; do not distribute it to untrusted users.
