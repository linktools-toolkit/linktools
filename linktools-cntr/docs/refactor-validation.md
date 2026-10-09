# Integration validation and deployment gates

Date: 2026-10-08.

## Scope and compatibility

The paired cntr and homelab changes use one public declaration contract:
`Integration`, `Integrations`, `Nginx`, `Flare`, `NginxSite`, `FlareCategory`, and
`FlareLink` in `linktools.cntr.integration`. Producers use `Nginx.site(...)` and
`Flare.public/bookmark(...)` to return one flat finite iterable of
declarations. Each declaration identifies its consumer; nginx sites carry stable
local IDs. Flare accepts ordered unnamed links and site-attached navigation.
See [integrations.md](integrations.md) for the authoring API and lazy URL helpers.

The producer authoring surface remains `configs`, `dependencies`, and
`integrations`. Native behavior belongs to the same `Container(BaseContainer)`:
`generates_config=True` explicitly enables `render_config`, `validate_config`
and `apply_config`, with optional `on_prepare_config`. There is no separate
consumer class, discovery contract or generated callback map. The manager's
`generated_configs` contains the installed owning containers themselves.
The `integration/` package owns shared declarations, factories and site
resolution; native lifecycle policies live on the asset container definitions.
Removed `config_sources`, exposure helpers and URL mixins are not compatibility
APIs. Domain configuration uses `Nginx.domain(container, name)` instead of
`BaseContainer.get_nginx_domain`.

The shared core orchestrates startup checks/hooks, resolved Compose capture,
image preparation, generated-input preparation/rendering, validation, bootstrap,
publication, application and rollback. Pull/build completes before
`on_prepare_config`; every final generated candidate is validated before restart
stops explicit targets. Pure `render_bootstrap(generation_id)` returns an
intermediate file mapping; the core reuses candidate validation, publication,
application and rollback rather than handing orchestration to a container.
Bootstrap application does not record final applied Compose snapshots.
Acknowledged bootstrap services can satisfy dependency availability and can be
restored after a cold final-application failure. On cold starts, unacknowledged
bootstrap services and unrelated stopped siblings are not rollback restore
targets. The service topological order preserves container, Compose and native
runtime edges, with `application_priority` breaking ready-node ties only.
`on_service_started` runs after each applied service and before dependents,
including native readiness checks where required. Preparation checks and
`BEFORE_START` hooks may include other running owners because their hooks affect
inputs; `on_started`/`AFTER_START` are restricted to the final application targets.
Planning describes this scope without executing callbacks or writing output;
it shares pure Compose/Dockerfile
serializers and service-ordering semantics with execution.

Complete candidate comparison allows a partial `up` or `restart` to apply
pending changes to other running services; unrelated stopped services stay
stopped unless required by runtime dependencies. Validated generated trees may
still be published for stopped owners without starting their services.

This pass starts from the paired published trees at main `f1fdab242fbd1ab89e8693173fb68f949fa7790b`
and homelab `d126d9bf6ea5ce6e374336588653619e10c71636`. Source-level compatibility
comparisons use main `4609177edc035a23c676b2bdf893e2930a110ae4` and homelab
`c918917e7a346ea223bb007659dec761705cf4a0`. No package version bump is included.
Both repositories must be updated together; their existing version constraints
do not distinguish this unversioned protocol change.

## Local check procedure

Use current source paths explicitly when the environment contains another
editable checkout. Run repository checks through `manage.py`:

```sh
PYTHONPATH="$PWD/linktools/src:$PWD/linktools-cntr/src" \
PYTEST_ADDOPTS='--ignore=tests/cntr/test_nginx_native_request_contract.py' \
python manage.py check linktools linktools-cntr
```

The excluded native request module requires separate target-environment
validation. Source checks, mock process tests, and deterministic template renders
do not launch Docker, issue certificates, or establish deployment acceptance.
Package build and artifact verification have not been performed for this pass.

## Regression coverage

- Declaration snapshots: mixed consumer types, local ID validation and duplicate
  rejection, one-time finite-iterator consumption, immutable structure, insertion
  order, disabled consumers, and lazy values that stay unread during collection
- URL factories: delayed configuration/host/site reads, immediate local-ID
  validation, missing and disabled values, per-producer site identities, and
  unchanged path/query composition
- Navigation: original 65 homelab and seven builtin links, site-attached before
  standalone links, standard category order, and the Authelia `/auth-admin` link
- Native templates: explicit namespaces, no removed snippet references, runtime
  Docker DNS, preserved prefix/capture/query routing, and complete header macros
- Lifecycle: restoration after failed nginx bootstrap, matching applied Compose
  snapshots, certificate replacement despite unchanged config IDs, aligned ACME
  cron configuration, explicit/implicit Compose dependency ordering, native
  container capability ownership, image-before-config preparation, dependency-safe
  priority ordering, pure bootstrap rendering through the shared candidate path,
  acknowledged bootstrap availability and cold rollback, per-service readiness,
  and post-start callbacks restricted to final application targets
- ACME build: the nginx image issues the full declared SAN set before deployment;
  offline preparation imports the built certificate and its account, checks keys/SANs,
  and retains previous versions without starting a new ACME challenge
- Candidate serialization: shared pure Compose/Dockerfile destinations and text,
  read-only collection, byte parity with writers, single model reads and rejection
  of invalid Compose values before destination creation
- Runtime arguments: literal-dollar decoding only at the raw Docker boundary,
  retaining correctly escaped persisted Compose files for later rollback
- Paired commands: MCP Playwright and MCP Push `show` handlers, including Push's
  optional channel argument

Nginx ACME issuance runs during Docker image build, not startup. Image tags
incorporate the declared domains, CA settings and base nginx tag, causing
changed SAN requirements to produce a new image before application. Runtime
preparation operates offline: it checks the persisted certificate, copies the
preissued image certificate and matching ACME account into a candidate version
only if the active certificate is insufficient, and verifies SAN coverage and
the private key. The selected `certs/live` symlink changes atomically and
rolls back on failure. The previous account remains preserved; newer images
bring their own versioned ACME state for subsequent cron renewals. The single-stage build reads DNS credentials through a BuildKit secret backed
by a private host file and runtime directory mount; the Nginx service environment
does not expose them.
Only ACME subprocesses receive those environment variables, and saved DNS
fields are removed after ACME invocation. The image still contains
certificate keys and ACME account state under `/opt/nginx-initial`;
do not publish it to an untrusted registry. Real CA issuance and image-key
distribution still require target-environment review.

## Remaining acceptance gates

| Area | Remaining target-environment validation |
| --- | --- |
| Native nginx | Full migrated homelab route set, HTTP/2 gRPC, SSE/WebSocket, and unusual custom headers |
| Authentication | Real Authelia sessions, OIDC flows, and alternate-authentication provenance assurance |
| SafeLine | Actual target images, Docker networks, request metadata, and forwarding behavior |
| Certificates | Real ACME issuance, persisted account reuse, changed SANs, renewal/cron reload, and live TLS certificate verification |
| Lifecycle | HTTP/WAF and HTTPS/auth cold starts, bind mounts, Unix health socket permissions, reload/rollback acknowledgements, and concurrent Docker operations |
| Persistent state | Legacy migration copy/mount behavior and real LDAP/password interoperability |
| Packaging | Built artifacts and artifact verification |

Earlier native nginx request results do not verify the changed paired templates.
The production health endpoint continues to use a Unix socket. No native
request, authentication-provenance, live Docker, or deployment check was rerun
for this pass.

LLDAP has no standalone upstream configuration-validation command. Its fixed
builtin TOML and derived password are staged without touching `/data`, then
service health is required after application. Changing a configured LDAP
administrator password does not reset an existing LDAP database password.
Legacy nginx migration preserves certificate/ACME/config backups without
replacing earlier backups, including existing stopped containers.

Compose `service_completed_successfully` handling remains a separate known
coverage gap; no current paired-repository caller uses it. It is not established
by the dependency-ordering regressions above.

**Local tests are not deployment acceptance.** Complete the remaining gates in
an authorized target environment before deploying the paired changes.
