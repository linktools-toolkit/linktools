# Container operation lifecycle

The public context is `OperationContext`. A container uses `on_starting` for
persistent initialization and `on_check` for validation. Derived configuration
paths are ordinary, optional container properties. `prepare_files` registers an
existing start hook rather than adding a configuration lifecycle callback.

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

- `context.compose_model` is the native Compose model with selected image inputs.
  Generated-file mount sources are declared directly by the template; the
  framework does not substitute mount paths after rendering
- `context.initial_runtime_state` owns observed services, image IDs and exact
  namespace bindings. Derived views do not introduce another state cache
- `context.service_models` (`AppliedServiceModels`) owns immutable current and
  previous per-service YAML snapshots. `previous_model(service)` returns an
  independent decoded copy; recording and restoring retain the disk format
- `context.prepared_dirs` records immutable trees materialized by start hooks.
  Successful applied snapshots retain the actual mount paths needed by running
  services and rollback; there is no mutable publication pointer

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

Only consumers whose generated-file mount changes join service application.
The new model must otherwise match its successful applied snapshot, including
referenced network/volume definitions. Pending environment, image, port or other
Compose changes require explicitly selecting that consumer when its generated
configuration also changes. Unchanged configuration leaves the consumer alone.
Actual running images are pinned without extra pull or build. Missing applied
snapshots require explicit selection. Unchanged consumers receive no application
callbacks. Additional consumers execute only the generated-file start hooks,
not their persistent initialization or source-download hooks.

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
2. Read related running consumers' declared models. Their cached configuration
   paths refer to content-addressed directories and register deferred file writes.
   Compare those models with successful applied snapshots to choose actual targets.
3. Explicit targets' container `BEFORE_START` hooks run normally. Additional
   changed consumers run only their generated-file hooks. Manager `BEFORE_START`
   runs once, followed by final model validation.
4. Prepare explicit images, then run `on_check` and `CHECK` for actual targets.
   All candidate files exist, but no service has been stopped or replaced.
5. Apply services through Compose, confirm readiness and record successful models.
   The source paths in those models identify the exact generated files used.
6. `on_started` and `AFTER_START` run for applied owners. A failure here reports
   that services were applied; it does not trigger deployment rollback.

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
repeated calls. Generated files are prepared through deferred start hooks without
replaying persistent initialization for collateral consumers.

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

Only containers that produce configuration need a path property. For example:

```python
from linktools.cntr.artifacts import prepare_files
from linktools.decorator import cached_property

@cached_property
def config_path(self):
    return prepare_files(self, self.render_configuration())
```

The Compose template declares `{{ container.config_path }}:/etc/app:ro` directly.
The property name is local to the container; the framework neither requires nor
looks up `config_path`. It is fixed for the command's integration snapshot, just
like the existing resolved-site and rendered configuration views.

Compose rendering resolves configuration for installed containers, including
installed consumers outside the selected service scope. Uninstalled consumers
are not rendered. A missing cached config provider may initialize its persistent
value at this boundary; an existing value is reused. The start hook writes the
already captured content without resolving configuration again. Read-only
Plan/Doctor resolution cannot initialize a missing cached value: it reports
the missing configuration rather than generating a credential or a fallback.

`prepare_files(container, files, mode=0o600, group=None)` copies the relative-name
to UTF-8-content mapping, computes its immutable directory and registers a
`BEFORE_START` callback using the existing hook registry. It performs no file
writes. The callback materializes those same bytes through `context.write_files`;
`context.file_path` or the declared property can locate them for validation.
Content, filenames, file mode, group and directory mode all contribute to the
path hash. Identical inputs retain the same path. Roots and nested directories
use `0755`; file permissions default to `0600`. A caller may supply `0640` and its
configured group. Existing immutable trees are checked for drift, never repaired
in place. Native checks validate the candidate before services are changed.

The successful applied model is the application baseline. Preparing files or
recording artifact provenance is not evidence that a service applied them.
Different services may safely retain different immutable generations; old paths
stay available for rollback. No configuration-revision label, shared `current`
symlink or extra revision getter is required.

Persistent secrets, databases, user assets and ACME account state are not ordinary
prepared configuration. They are not replaced by this file mechanism.

## Applying and recovering

Changed generated-file source paths or other Compose definitions recreate the affected service.
Unchanged definitions are not force-recreated. Docker Compose can still replace a
service when its locally selected image changes. Recreating a container does not
by itself rebuild its image or request a new certificate.

Build identity comes from the final Compose inputs. For a local `SourceContainer`
snapshot, the image preparer fingerprints its resolved versioned context, the
pending generated Dockerfile (or native Dockerfile/inline content), build options
and platform. Container authors do not implement `get_build_revision`. Nginx
uses its certificate revision in the native image tag.

Ordinary `up` reuses an existing image when no local source snapshot is present;
computing identity neither downloads source nor forces a build in that case.
A retained local snapshot built with the previous fingerprint scheme may require
one rebuild to acquire the current label. Subsequent unchanged runs reuse it.

Readiness follows effective Docker health checks and Compose dependency
conditions. Successful finite jobs are recorded as applied, not running.
Dependency/job readiness has no newly imposed fixed task deadline. A failure
blocks subsequent dependents.

The per-service resolved applied model is the sole recovery authority. Recovery
pins the pre-operation image ID and restores failed or explicitly stopped
services using their old immutable file paths. Existing recovery for changed
shared bind inputs also applies, including transitive consumers. New services
without a prior instance are stopped when required by recovery; confirmed
independent successes remain applied. Initial migration may read old saved
Compose files, but new operations do not write a second container-level snapshot.
Live namespace consumers are rebound after a retained provider changes, even if
an independent service later fails. During full-project recovery, restoring a live consumer may temporarily
start its previously stopped namespace provider with the old model and image;
the provider is stopped again after recovery. Interrupting deployment runs
recovery before propagating the interrupt; interrupting recovery itself stops
that attempt immediately and may leave it incomplete.

Generated directories referenced by current or previous successful service models
are retained. Unknown legacy running inputs are retained conservatively until an
explicit migration establishes their exact model and generated-file provenance.

A failed recovery reports both the operation and recovery errors. The framework
never claims to undo database or external-system changes.

## Built-in services

LLDAP, Authelia, Nginx and Flare retain persistent initialization in `on_starting`
and declare their immutable generated mounts through optional cached properties.
Authelia and its admin service reference the same generated directory; Redis does
not mount it. A partial failure can retain successful services because no shared
publication pointer is changed. Nginx rejects collateral changes requiring a new
certificate image; select it explicitly to prepare and build that image.
Authelia validates with its target image. If the main service is unavailable for
native validation, select Authelia explicitly rather than starting it implicitly.

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
