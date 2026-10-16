# Container operation lifecycle

The public context is `OperationContext`. A container uses `on_starting` for persistent initialization,
`on_integrating(context) -> None` for derived integration files, and `on_check`
for validation. A successful callback returns normally; failures raise exceptions.
`get_config_revision(service) -> str | None` exposes a prepared configuration
identity for each service; it does not prepare files or apply configuration.

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

- `context.compose_model` is the native Compose model with configuration-revision
  labels and selected image revisions. Deployment mount sources remain exactly
  as declared; native checks override mounts to read the staged candidates
- `context.initial_runtime_state` owns observed services, image IDs and exact
  namespace bindings. Derived views do not introduce another state cache
- `context.service_models` (`AppliedServiceModels`) owns immutable current and
  previous per-service YAML snapshots. `previous_model(service)` returns an
  independent decoded copy; recording and restoring retain the disk format
- `context.prepared_dirs` owns candidate immutable file trees. Publication switches
  `generated/current` only after validation and before service application.
  Successful applied snapshots and captured previous references retain the
  generations needed by running services and rollback

Image preparation has one orchestration entry point. The image preparer computes
build/pull requirements; the runner executes commands; `ComposeOperations`
orders them and owns warnings, checks and failure handling. Recovery separately
follows dependent actions, required stopped namespace providers and shared file
inputs because those relationships require different restore actions.

## Ordering and responsibilities

A named `up` or `restart` prepares images and initializes only explicitly
selected containers. It also generates candidates for related, already-running
integration consumers. Current declarations and stored producer provenance both
identify consumers, including removal of the final declaration. Stopped consumers
are not initialized or started.

Only consumers whose final configuration revision changes join service application.
Their previously applied service models are retained: pending environment, port,
image and other Compose changes are not applied. Their actual running images are
pinned, without extra pull or build. Missing applied snapshots or required image
changes require an explicit joint selection. Unchanged consumers receive no
before/check/after application callbacks. Required external services must already
satisfy their runtime conditions.

There is no additional global runtime-provider discovery or readiness gate.
Container dependencies and native Compose edges retain their existing meaning.
Nginx performs its own validation when it is applied; an unselected nginx
container does not impose checks on unrelated named operations.

An operation without names still resolves the full project, its required
providers and running declared consumers. Full-project generated-input metadata
retains producer names so removed declarations can update previous consumers.
Only real Compose service edges determine startup order; dependency cycles are
errors. The full Compose model and its dependency declarations remain intact in
both modes, and execution uses `--no-deps` with the resolved service list.

A named operation on a shared Compose service requires all contributing
containers to be named, so no unprepared contributor is silently applied. Native
Compose resolves their merged definitions. Managed build revisions follow the
contributing build owner; ambiguous build ownership fails before application.

Running outside consumers with `restart: true` edges or namespace bindings can
need joint application or recovery when a selected provider changes. Named
operations diagnose these relationships before image work and ask for the
consumer containers to be included explicitly. Even recovery cannot implicitly
start, stop or recreate an unselected service. Required providers in selected
services' previous models are checked as rollback prerequisites as well.

Compose resolves profiles before startup hooks. Full-project operations apply only
active services; explicit service selection enables its profiles through native
Compose. Disabled definitions remain available for orphan detection, while their
running containers and existing generated inputs stay unchanged unless a real
dependency action requires rebinding. A plan that cannot resolve native profiles
reports the unresolved selection rather than guessing startup commands.

The operation then runs:

1. Explicit targets' `on_starting` initializes persistent state and build inputs.
2. Targets and related running consumers' `on_integrating(context)` generate
   immutable files. This callback must not initialize secrets or mutate services.
   Each consumer exposes its prepared generation through
   `get_config_revision(service)`. The framework compares revision labels against
   successful applied snapshots.
3. Actual application targets' container `BEFORE_START`, then manager
   `BEFORE_START` once. The final Compose model is read after these hooks.
4. Explicit image preparation, then `on_check` and `CHECK` for application targets.
   Native checks read candidates through explicit mount overrides. No service
   has been stopped or replaced and no `current` reference has changed yet.
5. Switch each affected consumer's `generated/current` reference atomically, then
   apply services in Compose dependency order. Confirm readiness before recording
   successful applied models. Services sharing one consumer generation form one
   recovery group.
6. `on_started` and `AFTER_START` for applied owners. A failure here reports that
   services were applied; it does not trigger deployment rollback.

`restart` performs its explicit stop only after step 4. Required providers that
were not explicitly selected are not included in the stop set. `down` uses the stop callbacks after loading preparation and performs no
start preparation, build or native check.
Status reads metadata and actual state without invoking preparation callbacks.
Named operations diagnose required outside actions before image refresh or service
mutation. Explicit joint selection is required even when rollback could need to
rebind a running dependent. Inspection and readiness checks do not imply that an
existing service is restarted or recreated.

`on_prepare()` runs after config registration. Named `up` and `restart` prepare
only explicitly named owners before reading their service declarations. Full
starts, `down`, and `compose` retain project-wide preparation. The standard
`exec <name> up/restart` entry delegates preparation to the same named operation;
other exec commands retain their existing preparation entry. Exceptions propagate
before the requested runtime operation starts. Custom callbacks must tolerate
repeated calls. Generated integration files belong in `on_integrating(context)`,
which also runs for related running consumers without their initialization hooks.

Metadata-only planning, status, doctor and config commands do not call
`on_prepare()`. A hook that changes declarations can therefore change the actual
execution selection compared with a read-only plan. Keep declarative service and
dependency definitions side-effect free when they need to be visible in plans.
The existing seven `HookPhase` values are unchanged.

## Template paths and deferred actions

Template getters return data without executing filesystem or runtime changes.
For a shared directory, the consuming template uses `path | mkdir | chown(docker_user)`.
These filters register idempotent `BEFORE_START` actions on that consumer, so
previewing or repeatedly rendering a cached provider's paths does not execute them.
A partial start prepares the selected consumer's mounts even when the path provider
has no service of its own. `DOCKER_USER` is the shared project configuration;
`chown` is non-recursive unless explicitly requested.

The old `start_hooks`/`stop_hooks` list views are removed. Custom deferred actions
use `add_start_hook(key, callback)` / `add_stop_hook(key, callback)` or the existing
`hooks.register(...)` API. Register work on the owner whose operation needs it;
do not append actions to an unrelated provider from a cached data getter.
Exceptions propagate before application. Persistent shared directories are not
rollback artifacts and are not deleted if later validation or startup fails.

## Preparing files

Use `context.write_files(self, files, mode=0o600, group=None)` once during
`on_integrating`. `files` maps relative names to UTF-8 text. The call returns an
immutable host directory and does not publish it to running services. Generated
files remain owner-only (`0600`) unless the producer explicitly supplies another
mode. For a non-root reader, use `mode=0o640, group=<container group>` and make that
group part of the service's effective or supplementary groups. New generation
roots and nested directories are explicitly `0755`, independently of the CLI
umask; this directory policy is part of the generation identity. Reuse checks
both file and directory permissions and rejects drift rather than repairing an
active generation. A group change needs no privilege when the CLI owns the file
and the group is its effective primary or supplementary group; other changes
retain the normal privilege path.

Compose templates declare the runtime paths they need, usually under
`APP_PATH / "generated/current"`. The framework does not rewrite deployment bind
sources into hash-directory paths. A native check uses explicit `mount_overrides`
to read the corresponding staged candidate, or `context.file_path(self, name)`
for a local format check. Check callbacks do not overwrite prepared inputs or
publish the candidate.

Store the directory name returned by `context.write_files` as the candidate
revision. Return that value from `get_config_revision(service)` only for services
that consume this generation; return `None` for other services or before a
candidate has been prepared. Clear operation-local revision state in
`on_prepare`. The getter must be a read-only query and its result must not contain
configuration values or secrets. The base implementation returns `None`.

The framework adds `io.linktools.cntr.config-revision` to the effective service
model. This is a configuration identity, independent of
`io.linktools.cntr.build-revision`: it does not request an image pull or build.
The successful applied model is the baseline; preparing a generation or writing
its artifact-index entry is not evidence of successful application.

All services consuming the same generated tree share its revision, even when
one service mounts only a single file. This keeps publication and failure
recovery atomic for that consumer. Services that do not read the tree receive
no configuration revision. Missing inputs or modified immutable trees cause an
explicit failure before application.

Persistent secrets, databases, user assets and ACME account state are not ordinary
prepared configuration. They are not replaced by this file mechanism.

## Applying and recovering

Changed configuration-revision labels or other Compose definitions recreate the affected service.
Unchanged definitions are not force-recreated. Docker Compose can still replace a
service when its locally selected image changes. Recreating a container does not
by itself rebuild its image or request a new certificate.

Readiness follows effective Docker health checks and Compose dependency
conditions. Successful finite jobs are recorded as applied, not running.
Dependency/job readiness has no newly imposed fixed task deadline. A failure
blocks subsequent dependents.

The per-service resolved applied model is the sole recovery authority. Recovery
pins the pre-operation image ID and restores the failed service, any explicitly
stopped services not yet reapplied, and every affected service sharing the same
consumer configuration tree. Existing recovery for services connected through
changed shared bind inputs also applies, including transitive consumers.
Restore the group's previous `current` reference
before recreating its services from their old models. A partly successful group
must not keep mixed old and new configuration revisions. Newly started related
services without a prior instance are stopped. Independent successful groups are
retained. Initial migration may read old saved Compose files, but
new operations do not write a second container-level applied snapshot.
Live namespace consumers are rebound after a retained provider changes, even if
an independent service later fails. During full-project recovery, restoring a live consumer may temporarily
start its previously stopped namespace provider with the old model and image;
the provider is stopped again after recovery. Interrupting deployment runs
recovery before propagating the interrupt; interrupting recovery itself stops
that attempt immediately and may leave it incomplete.

`generated/current` is the publication reference used by the declared runtime
mounts. Its switch occurs after CHECK and before `up`; switching it alone does not
establish a successful service application. On failure, restore the previous
reference together with the group's old service models. Only confirmed success
advances the applied revision baseline. Retain the previous generation until it
is no longer needed for running services or recovery.

A failed recovery reports both the operation and recovery errors. The framework
never claims to undo database or external-system changes.

## Built-in services

LLDAP and Authelia retain persistent initialization in `on_starting` and stage
generated files in `on_integrating`. Nginx and Flare likewise separate
initialization from generated integration files. Each built-in clears its
candidate revision in `on_prepare` and exposes it only for services reading the
generated tree.
Nginx rejects collateral integration changes needing a different certificate image
revision; select nginx explicitly to prepare that image.
Authelia validates with its target image. Its main and admin services share the
configuration-tree revision and recovery group; Redis has no configuration
revision. Even an ACL-only change advances both readers' generation. If the
Authelia service is unavailable for native validation, select Authelia explicitly
instead of starting it implicitly for another container's operation.

Flare prepares navigation files with its configured group access and validates
the generated YAML. Host UID/GID values are not used to infer container access,
which depends on the runtime and its identity mapping. Initial
migration copies other legacy application assets to `runtime-app`, leaving the
legacy directory available to an old Compose model. Application uses normal
file mounts and service recreation, not migration callbacks around Compose.

Ownership hooks accept numeric IDs without requiring host account names:
`chown(path, user, recursive=False, *, group=None)` and the equivalent template
filter `path | chown(DOCKER_UID, group=DOCKER_GID)`.
The template filter leaves an omitted or `None` user as a placeholder: it returns
the path without registering an ownership hook or resolving default IDs.
Explicit users retain their existing behavior: a named user with no group keeps
its existing primary-group behavior; a numeric user with no group preserves the
file's group. The third positional argument remains `recursive`. Numeric zero
is a valid owner. The lower-level `runtime.chown` requires an explicit user.
Existing bare `chown` calls on persistent-directory mounts remain placeholders.
When choosing an explicit owner, ensure the CLI identity can still maintain paths
it needs to write.
Do not apply ownership hooks to immutable generated inputs; set their access
permissions through `write_files` instead.

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
