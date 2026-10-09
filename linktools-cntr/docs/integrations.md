# Container integrations

`integrations` returns a flat list of `Integration` declarations. Each declaration
identifies its consumer. `NginxSite` identifies a site by
`(producer.name, local_id)`; the local ID is a nonempty string. Consumers
read the complete installed declaration snapshot, even for partial operations.
Unknown consumer names are errors. Known optional consumers that are not
installed do not consume declarations.

```python
from linktools.cntr import Flare, Integrations, Nginx
from linktools.cntr.integration import load_port_url


@cached_property
def integrations(self) -> "Integrations":
    return [
        Nginx.site(
            self.get_config_later("APP_DOMAIN"),
            local_id="web",
            proxy="http://app:8080",
            auth=None,
            auth_bypass=(r"^/public/",),
            waf_bypass=(),
            expose=Flare.public("App", "apps", "Application"),
        ),
        Flare.bookmark(
            "App direct", "apps",
            load_port_url(self, "APP_PORT", https=False),
            category="container",
        ),
    ]
```

`Nginx` and `Flare` provide the declaration factories. `Nginx.site(...)` returns
a `NginxSite`; `Flare.public(...)` and `Flare.bookmark(...)` return a `FlareLink`.
The `integration/` package owns these factories and public declaration types, also
re-exported from `linktools.cntr`. `Integration` declares public `consumer` and
`local_id` metadata. A declaration with `requires_local_id=True` must supply a
nonempty local ID, including nginx site declarations.
`NginxSite.consumer` is `"nginx"`; `FlareLink.consumer` is `"flare"`. New
consumer-specific subclasses set their own `consumer` name and optionally a
`local_id`. `Integrations` is the Python 3.6-compatible alias
`Iterable[Integration]`: return a finite list, tuple, or iterator of declarations.
The former consumer-keyed mappings and nested named mappings are not accepted.

`NginxSite.local_id` defaults to `"web"`. Give additional sites explicit IDs,
for example `Nginx.site("api.example.com", local_id="api")`. IDs must be nonempty
strings and unique for each `(producer, consumer)` pair. Different producers or
consumers can reuse the same ID. Flare links have no local ID and do not need
invented names.

The manager freezes each installed producer's declaration structure once per
command into a read-only producer mapping of declaration tuples, including
consuming a generator once. Mixed declaration order is preserved in the snapshot.
Iterables must be finite. If `integrations` is cached and its snapshot can be
rebuilt, return a reusable list/tuple or provide a fresh iterator for the rebuild;
do not reuse an exhausted generator.
It checks consumer names, local IDs and the `Integration` type without
resolving lazy fields or URLs. Consumers validate their own declaration content.
The generic `iter_integrations` filters by the declaration's consumer and yields
`(producer, declaration.local_id, declaration)`. It returns only explicit
declarations for an installed consumer; unnamed declarations have `local_id=None`.

`NginxSite.expose` optionally attaches a `FlareLink` for navigation. Omitting
its URL lazily inherits the resolved site's URL. Explicit `None` or `""` disables
the link; an explicit URL stays unchanged. An omitted URL on a standalone link
has no site to inherit from and is skipped. The link's category is presentation
metadata, not an authentication policy; there is no `public` site boolean.
Sites without `expose` create no navigation.

Standalone `FlareLink` values in the list supply independent navigation entries.
Use `Flare.public(name, icon, desc, url)` for application entries; their
descriptions are included in `apps.yml`. Use
`Flare.bookmark(name, icon, url, category="tool")` for bookmarks, which do not
need an application description. A custom string category creates a bookmark
group with that ID and title. The standard IDs `private`, `container`, and
`other` use their predefined titles and orders; omitting `category` uses `other`.

Pass a `FlareCategory` created by
`Flare.category(name, desc=None, *, apps=False, order=None)` instead of a string
to customize a bookmark group's title or order. Custom categories default to
their ID as the title and order 100; smaller order values come first. Calling
`Flare.category` with only a standard bookmark ID selects its predefined group.
Explicit category metadata determines the output area and ordering.
For example:

```python
from linktools.cntr import Flare

tools = Flare.category("tool", "Tools", order=5)

Flare.public("App", "apps", "Application", "https://app.example.com")
Flare.bookmark("Docs", "book", "https://docs.example.com", category=tools)
Flare.bookmark("Team", "account", "https://team.example.com", category="team")
```

`apps=True` writes links to `apps.yml`; other categories write to `bookmarks.yml`.
The standard public category declares `apps=True`; the standard bookmark
categories declare orders 10 (`private`), 20 (`container`), and 30 (`other`).
Custom bookmark categories default to order 100 and can override it to appear
before or between the standard categories. Equal orders keep first-seen category
order. A custom app category created with `Flare.category(..., apps=True)` is
callable as `category(name, icon, desc, url)` to create application links.
`Flare.bookmark` requires a bookmark category; app categories are rejected.
The old `ExposeCategory` and `ExposeLink` names are removed; use `Nginx` and
`Flare` factories for declarations, and `FlareCategory` and `FlareLink` for value
types. The former `exposes` property
and `self.expose_*` helpers are removed without aliases or fallbacks.
Links keep their category, name, icon, description and lazy URL; direct-port,
external and non-HTTP links do not need an nginx site. Flare preserves container
`order` (and snapshot order for ties). Within each producer, attached links follow
nginx declaration order, then independent links follow their declaration order.
App links keep traversal order across all app categories;
`FlareCategory.order` applies to bookmark categories. Links within each bookmark
category keep traversal order.
Flare merges these two inputs itself; the manager's `iter_integrations` returns
only explicit declarations. It skips empty URLs and rejects conflicting
descriptions, output areas, or orders for the same category ID. A site
without navigation still generates a proxy. The former registration arguments,
`write_nginx_conf`, `load_exist_nginx_url`, and `append_ssl_domains` are removed.
When migrating a legacy `write_nginx_conf(..., auth_enable=False)` call,
set `Nginx.site(..., auth=False)` explicitly to retain its no-auth behavior.
`auth=None` now inherits `NGINX_AUTH_ENABLE`; HTTP-only and public
sites must use `auth=False` when global authentication is enabled.
Migrate callers and templates together; there is no compatibility wrapper.

## Lazy URL references

Declare domain configuration with `Nginx.domain(container, name=None)`, which
returns a lazy configuration provider. For example:

```python
APP_DOMAIN = ConfigField(provider=Nginx.domain(self))
```

It keeps the shared nginx root-domain, wildcard and disabled-provider behavior.
The former `BaseContainer.get_nginx_domain` method is removed.

Import the URL factories from `linktools.cntr.integration` and pass the owning container:

```python
from linktools.cntr.integration import load_config_url, load_nginx_url, load_port_url

load_config_url(self, "APP_URL", "ui", queries={"mode": "compact"})
load_port_url(self, "APP_PORT", "ui", https=False)
load_nginx_url(self, "web", "ui")
```

Each returns the existing lazy URL proxy. Config URLs and port URLs stay empty
when disabled. Ports must satisfy `0 < port < 65535`; the host is not read until
the port is valid. nginx local IDs are checked immediately, while the
`(container.name, local_id)` lookup waits until the proxy is read. Unknown site
IDs fail, and declared disabled sites or absent nginx resolve to an empty URL.
Path joining, query encoding and literal template placeholders remain unchanged.
These factories never register hooks or write configuration.

`BaseContainer.load_config_url`, `load_port_url`, `load_nginx_url`, and the old
exposure mixin are removed; no URL mixin or resolver wrapper replaces them.
General Compose/Dockerfile templates expose the public URL factories as `urls`,
alongside `utils`, so they can use `urls.load_nginx_url(container, "web")`. Native nginx
templates retain only the explicit context described below.

## Site values

- `server_name=""` disables the site and unrelated secret/config resolution
- `default=True` makes this site the native default server on its HTTP, HTTPS
  (when enabled), and WAF (when enabled) listeners, regardless of its hostname.
  Enabled defaults must not share a listening socket. If none is declared, nginx
  adds one built-in fallback site. `server_name="_"` alone no longer requests
  default behavior; migrate catch-all declarations to `default=True`
  (this is separate from `NGINX_ROOT_DOMAIN`)
- `https`, `waf`, and `auth`: `None` inherits, `False` disables, and `True`
  requires the global capability; missing providers fail closed
- Browser authentication requires HTTPS
- `auth_bypass` and `waf_bypass` independently match the normalized main-request
  path without query; no application path is implicitly trusted
- `auth_headers` is literal data, injected only after actual successful
  authentication; bypass and explicit native auth-off preserve client tokens
- `auth_rule` is an optional native Authelia rule; a literal domain is filled in
  when omitted. Pattern domains need an explicit native domain/domain_regex
- `oidc_redirects`: absolute URI, empty string for the exact public URL, or a
  root-relative path. Protocol-relative URLs and fragments are rejected
- Only a single literal hostname implies a public URL. Pattern/default sites
  need explicit `url`; navigation placeholders such as `{{port}}` remain literal
- `cert_domains` declares additional certificate names; `vars` owns business
  template variables

Authelia exposes a read-only `oidc_client` mapping with `client_id`,
`client_name`, `client_secret`, `issuer_url`, `authorization_url`, `token_url`,
`userinfo_url`, `user_identifier`, and tuple `scopes`/`redirect_uris`. Consumers
must not mutate or restore historical derived ACL/OIDC settings.

## Native nginx templates

A custom template owns the server's business directives and locations, not its
listeners, TLS, authentication endpoint or WAF forwarding. Context contains
`site`, `container`, `nginx`, `config`, and `vars`, plus `route_auth=True` only
when sharing a hostname requires location-level authentication. Read environment
values with `config.get("KEY")`. Undefined fields fail rendering.

Jinja names start with `local/` (the entry template's directory tree) or
`nginx/` (builtin templates). Business templates render once. Native nginx
`include` loads generated output and is distinct from a Jinja include. Data is
not rendered again; use Jinja comments to disable logic and raw blocks to emit
another template language.

```jinja
{% from "nginx/headers.j2" import proxy_headers with context %}
location / {
    {{ proxy_headers({"Host": "$host", "Authorization": "$http_authorization",
                      "X-Forwarded-For": None}) }}
    set $upstream "http://app:8080";
    proxy_pass $upstream;
}
```

`proxy_headers` and `grpc_headers` share one default header definition and emit
complete same-level sets. On shared hostnames with different authentication
requirements, use `route_authorization()` (also from `nginx/headers.j2`)
inside each native location, or write equivalent native nginx directives.
The authorization macro enables that route's declared authentication, or turns
it off for an unauthenticated route. A location without an explicit override
inherits the server's deny-by-default check. Native nginx validates directive
syntax; generated text is not searched for a particular directive spelling.
Explicit native `auth_request off` continues to override inherited protection
for intentional login endpoints; do not call `route_authorization()` there.
Override names are case-insensitive; values are native
nginx complex values, not directives or prequoted strings. `None` or `""` emits
an empty value so nginx suppresses that header. Adding a separate native
`proxy_set_header` in a child location stops nginx from inheriting the parent's
whole header set; call the macro in every location that needs customized headers.

`Host` and ordinary `Authorization` can be overridden as above. If a header is
explicitly configured in `auth_headers`, it is instead an authenticated credential:
its override is rejected to preserve the successful-authentication requirement.
`X-Auth-*` identity headers and `X-Proxy-Original-*` internal metadata are reserved.
The WAF forwarding location sets the latter; application and auth locations strip
them. Original request values are `$original_scheme`, `$original_host`,
`$original_uri`, `$original_method`, `$original_client_ip`, and `$original_port`.

Generated output consists of `nginx.conf` and one self-contained
`sites/<site-id>.conf` per effective hostname and listener group (including
`sites/default.conf` when needed). Multiple declarations with the same
hostname, HTTP port and compatible HTTPS/WAF listener and WAF policy share one
server and contribute their distinct, once-rendered business locations. Literal
and wildcard hostnames are compared without case. Per-route authentication and
bypass settings may differ, using the explicit location authorization macro
above. Their producer/local IDs and per-declaration template variables remain
independent. Incompatible server policies are rejected; duplicate native
locations are rejected by `nginx -t`, never silently overwritten.
Read the root for shared request maps and the health listener; read a site file
for its maps, listeners, WAF path, authentication endpoint, and business locations.
Business templates are evaluated once and their text is embedded without another
Jinja pass. There are no generated per-site auth/business include directories.

For custom integrations migrating older generated names, replace `$cntr_*`
request references with the corresponding `$original_*` values above. Private
WAF metadata is now `X-Proxy-Original-*`, the internal auth URI is
`/_internal/auth`, and reload acknowledgement uses `/run/nginx-health.sock`
with `/health`. Producer and consumer must use the same generated configuration;
there are no old-name aliases. CLI names are unchanged; URL-helper imports migrate as described above.

Custom `auth_request off`, `return`, rewrite and native location
selection retain nginx semantics. An early `return` is not protected by the
access phase. Preserve URI replacement and captures when migrating static
upstreams to Docker runtime DNS.

## Selection, publication and migration

`dependencies` defines required installed dependencies and startup order
for an explicitly selected container. A different running service selected only
for configuration reconciliation follows its own Compose service dependencies,
without automatically starting the other dependencies of its owning container.
An enabled nginx site also needs its installed nginx runtime provider when the
producer is explicitly selected. Flare navigation is optional and never creates
a startup dependency. Explicitly selecting Flare still starts it.

Every `up` or `restart` resolves the full installed integration snapshot and
Compose model, then stages and validates generated candidates for the selected
services, their dependencies, and previously running services requiring
synchronization. Generated owners unrelated to the operation and currently
stopped are prepared when they are next selected or required. Resolved snapshots preserve environment values from `.env`,
`env_file` and Compose interpolation for comparison and rollback. Shared top-level
network/volume changes conservatively invalidate running service models.
Explicit targets and their runtime dependencies are ensured running. Other
services are updated only when their configuration changed and they are already
running; stopped sibling services are not started by configuration reconciliation.
A partial command can therefore apply pending changes to other running services.
If an ordinary service first started by this operation fails to apply or become
ready, that service is stopped and its newly written applied snapshot is
reverted; any originally running sibling remains untouched. If cleanup itself
fails, both the original and cleanup errors are reported.
When no resolved snapshot exists yet, a running service is reconciled once to
establish it; rollback uses the previous saved Compose file where available.
Historical external environment-file contents cannot be recovered retroactively.
Rollback pins each restored service to the image ID observed before the operation;
pulling or rebuilding a mutable tag must not change its restored image. A running
instance without an identifiable original image or saved Compose model is rejected
before replacement, including when a previous generated directory exists.
Acknowledged cold-start bootstrap fallbacks do not require an earlier image.
Saved Compose content executed from temporary files retains the original project
base directory with `--project-directory`; multi-file order and old model
dependencies remain authoritative. Only the affected services are restored.
Normal Compose dependency readiness has no implicit 30-second total deadline;
nginx config/TLS acknowledgement and explicit deadlines remain independently bounded.
A `depends_on` entry with `required: false` does not force its unavailable
provider into startup or image preparation. When that provider is running or
explicitly selected, the declared readiness condition still applies during
normal application and rollback. Starting dependencies continue to be observed;
known unhealthy or failed optional dependencies log a warning and do not block.
Inspection failures and required dependency failures remain errors.
A running generated-config consumer such as nginx expands newly required native
providers when its generated candidate changes; an independent Redis sidecar
does not inherit its owner's unrelated provider dependencies.
`restart app` stops only explicit targets, after all final and required bootstrap candidate validation passes.
For batched restarts, an application or post-stop hook failure restores any
previously running explicit targets that were stopped but have not yet completed
their application. If Compose stop reports an error, actual runtime state
determines which targets were stopped before the failure; still-running targets
are not replaced. The failed application attempts its own rollback; if that fails, the failed
service rejoins the still-pending restore set. The old dependency graph orders
the combined recovery so an old provider is restored before its dependent.
Previously confirmed independent applications are not silently reverted.
The observed partial stop state updates the running-state cache, while still
running sibling services or replicas preserve the owner's running marker.
Recovery uses original image IDs and saved models. A failure during runtime
inspection or restoration reports both the operation error and recovery error.
Startup callbacks for selected owners follow the dependency-ordered service
scope, including native runtime providers.
Preparation covers running generated-config consumers and their potential
native providers: `CHECK`, `on_starting` and `BEFORE_START` run before the
final Compose model is resolved, even if an eventual candidate is unchanged.
Application remains limited to the final changed-service selection, and
`on_started`/`AFTER_START` use only that final application scope. Independent
sidecars do not expand their owner's native providers. Snapshots are captured
after startup hooks, so hook-prepared environment files are included.
The plan reports the full reconciliation scope and defers runtime-dependent
update decisions until execution.

Container authors do not declare `config_sources` or an integration startup
policy. OIDC/template cross-container reads are captured by complete candidate
rendering. Aggregate consumers read the complete installed snapshot, including
removal of a producer's final declaration. No historical dependency graph or
runtime configuration-read tracking is needed.

Authelia reuses its resolved site URL for OIDC and session endpoints, joins
`settings` to that URL for default redirection, and derives the admin origin and
authority separately. Its cookie domain remains `NGINX_ROOT_DOMAIN`.
`ResolvedSite.url` requires an explicit URL when the server name is nonliteral;
metadata consumers may request `get_url(default="")` to leave only an absent
concrete identity unset. Invalid explicit values and configuration errors still
raise, and generated Authelia identity still requires HTTPS.

Producers declare their inputs through `integrations`. The owning
`Container(BaseContainer)` also implements its native runtime requirements,
generated configuration, bootstrap and readiness behavior. The loader discovers
one `Container` class; there is no separate consumer object or callback map.
Container construction and declaration collection must remain side-effect free.

```python
from linktools.cntr import BaseContainer


class Container(BaseContainer):
    generates_config = True
    application_priority = 0
    bootstrap_services = ()

    # Implement render_config, validate_config and apply_config here.
    # Override on_prepare_config when native inputs need preparation.
```

Set `generates_config=True` explicitly when the container owns generated files.
Overriding a method alone does not opt into generation. The manager's
`generated_configs` is a read-only mapping of installed names to those same
container instances. It does not construct an adapter or another lifecycle owner.
Implement these methods on the container:

- `on_prepare_config(context)` prepares native inputs after required image
  pulls/builds have finished. It is distinct from the existing zero-argument
  `on_prepare()` used during execution setup
- `render_config(generation_id)` returns a relative-path-to-text mapping for an
  immutable candidate tree
- `validate_config(context, candidate)` validates staged output before publication.
  All final and required intermediate bootstrap candidates are validated before
  any explicit restart target is stopped
- `apply_config(context, candidate, services)` applies the published candidate to
  the selected services and confirms it was loaded. It is also used to restore a
  previous generation on rollback; a successful reload command alone is not an
  acknowledgement
- `generation_services` defaults to all services of that container and may be
  overridden when an independent sidecar does not consume generated files.
  For example, Authelia's Redis sidecar is not a generated-config consumer
- `rollback_config(context)` restores native migration state when no prior
  generated version exists; the core restores the saved Compose service model

Other native capabilities do not require generated files:

- `get_runtime_requirements(required)` receives the selected container names and
  returns a provider-name-to-service-names mapping. It must be safe during planning
- `application_priority` defaults to zero; lower values win only among services
  whose dependency edges are already satisfied
- `on_service_started(context, service)` confirms native readiness after each
  service's application and before its dependents are applied. The default is a
  no-op, so containers with additional readiness requirements must implement it

A generated owner may declare `bootstrap_services` and implement the pure
`render_bootstrap(generation_id)` method, returning a relative-path-to-text mapping
for an intermediate configuration. This method does not start services or publish
files. The core creates the bootstrap candidate and uses the same
`validate_config`, publication, `apply_config`, readiness and rollback path as
for final candidates. Bootstrap application does not record a final applied
Compose snapshot; that is recorded only after final application succeeds.

An already-running bootstrap service must pass its health check. Otherwise the
core applies and acknowledges the intermediate configuration. Only acknowledged
bootstrap services count as available while ordering the complete application.
An ordinary running dependency is still applied in dependency order before its
dependents. On a cold start with no previous generation, a successfully
acknowledged bootstrap becomes the fallback for a failed final application.
Bootstrap is never treated as a pre-operation generation when replacing an old
running instance. Rollback restores previously running services and cold-start
bootstrap fallbacks, and stops newly started services whose deployment did not
complete; it must not start unrelated stopped siblings. During first migration,
when no previous generated version exists, native state (including the nginx
certificate pointer) and previously running services are restored using the
captured prior Compose model instead of treating the missing generation as
evidence that no old service exists. A failed, unacknowledged
bootstrap does not make its service a rollback restore target.
`generation_label(service, generation_id)` controls the generated service marker;
its default marks the same-named service of a generated owner.

Startup follows one orchestration path:

1. Run startup checks, `on_starting` and registered pre-start hooks for the
   preparation scope, which can include other running owners. Then resolve the
   authoritative Compose model so hook-prepared environment files are included
2. Prepare images for the explicit deployment scope and changed Compose
   services. Prepare and render candidate files for selected generated owners
   and running services which actually consume them. A stopped, unrelated
   generated service is not prepared merely because its sidecar is running
3. Reconcile generation changes, prepare any newly required images, then
   validate the final candidates and any required intermediate bootstrap candidates
4. For restart, stop only the explicit targets after successful validation
5. Health-check already-running bootstrap services or publish and apply the
   validated intermediate configurations through the core's candidate path
6. Apply services in topological order across container, Compose and native
   runtime dependencies. Priority breaks ready-service ties, never dependency
   edges. Publish generated candidates before applying their services and confirm
   per-service readiness before moving to dependents
7. Publish validated candidates for generated owners with no selected services
   without starting those services, then run `on_started`/`AFTER_START` only for
   the final application targets

Planning shares dependency and service-ordering semantics but does not execute
these callbacks, prepare images or write candidates. Compose and Dockerfile
candidate paths and text use the same pure serializers as execution's writers.

The `integration/` package owns declaration types, factories and shared site
resolution. Native implementations live on the container definitions in
`assets/containers/*/container.py`. nginx owns runtime provider requirements,
certificate preparation, bootstrap, native template validation, reload
acknowledgement and validation diagnostics. Flare owns category grouping,
navigation generation and application. Authelia and LLDAP own configuration
generation and readiness; SafeLine owns its management readiness.

The core owns generic dependency closure, Compose execution, candidate
publication and rollback. `ContainerManager.nginx_sites` only caches and
delegates site resolution to `Nginx.resolve_sites`; it contains no nginx
validation policy. There is no separate generation class or exported generation
module. This navigation and synchronization contract supersedes the earlier
Site protocol's separate `exposes` and `config_sources` properties. Migrate both
repositories together; there is no fallback alias.

Generated configuration uses a stable mounted parent, immutable generation
folders and an atomic `current` link. A published symlink is not proof that
the running service acknowledged the generation: existing running consumers are
checked against their actual runtime version and reconciled if out of sync.
Native nginx checks its existing generation health endpoint; ordinary generated
services check their runtime generation marker. Final candidates are rendered
and validated before restart stops a target. nginx loads bootstrap health/rejection config
when starting without a serving process; authentication/WAF providers become
ready before the complete nginx config is activated. nginx health returns the
loaded generation ID, and failed application restores the previous generation.
After the entire operation succeeds, generated-config cleanup first verifies
the running consumers; it retains unconfirmed generations instead of deleting
directories that a running service might still use. Confirmed cleanup retains
the active generation and the preceding generation known to that operation,
deleting older and abandoned version directories plus their artifact index entries. Failed
operations keep their staged files for diagnosis; a later successful operation
can discard them. This cleanup never touches certificate or ACME account storage.
Legacy restore inputs are resolved and checked before stopping a previously
running service, using only its old Compose files and their actual dependencies
and shared resources. An unrelated stopped owner's damaged configuration does
not block recovery. Per-service comparisons project only the service's
referenced networks, volumes, secrets and configs; saved restore snapshots
still contain the complete old project.
Cross-service state is not an atomic transaction; failures remain command errors.

nginx issues and installs its initial certificates while building the image, using
`ACME_SERVER` (default `letsencrypt`), optional `ACME_ACCOUNT_EMAIL`, and the
selected DNS API. DNS credentials are written directly as Dockerfile `ENV`
values for initial issuance and retained in the image. DNS credential values must
be single-line; CR or LF characters fail rendering with a clear error. Compose
also supplies the configured values as runtime environment variables for cron renewal. ACME account
and domain configuration is preserved with the initial certificate state.
Images and build contexts contain DNS credentials, certificates, private keys and
ACME account state: protect them and the build cache as secrets; do not publish them.

Deployment seeds only empty certificate/ACME mounts from the image, then validates
expiry and domain coverage offline. Existing mounts (including account keys and
renewed certificates) take precedence. Missing, expired or incompatible persisted
certificates fail validation; deployment does not request replacement certificates
from a CA. Build a fresh image and deliberately reconcile persisted state when
changing certificate domains. Legacy container files are backed up before mount
migration. The ACME client lives in `/opt/acme`, outside the persisted config mount.
An explicit daily cron uses the persisted config and a reload script that validates
the active generated configuration before reloading; legacy reload commands are
updated without changing their account or certificate keys. HTTPS-disabled images
skip ACME installation, issuance, DNS environment variables and cron.

Deployment migration must be coordinated with all external repository callers:

1. Back up existing generated config and record the matching repository commits,
   image IDs and SafeLine origin target. Preserve settings, keys and databases
2. Migrate external Python declarations, native templates and OIDC consumers
3. Recreate nginx for the stable generated-parent mount change
4. Point this integration's SafeLine origin to
   `http://nginx-origin:<NGINX_WAF_PORT>` and preserve all five `X-Proxy-Original-*` headers
5. Validate real Docker, SafeLine and Authelia/OIDC behavior before deployment

Only SafeLine's exact `.254` socket source is trusted at origin. nginx uses the
`.253` network address and `nginx-origin` alias; the origin port is not published.
Applications and Authelia never receive internal transport headers. The private
Unix health socket must work in the target image/environment.

LLDAP keeps `/data` databases and keys; its derived TOML and password use a
separate generated tree. Updating the password file does not reset an existing
LDAP administrator password. Authelia keeps `/config` databases and notification files; generated YAML and
the final LDAP password live under `/generated/current`. Existing random secrets
are retained. Flare preserves its user config and backs up replaced generated
navigation files. Rollback restores matched code/mounts/generated config and the
previous SafeLine origin; it must not overwrite newer business data or rotate
credentials. No version bump identifies this development protocol: both
repositories must be paired explicitly.
