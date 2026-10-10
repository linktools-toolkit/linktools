# Container integrations

`integrations` returns a flat list of `Integration` declarations. Each declaration
identifies its consumer. `Nginx` identifies a site by
`(producer.name, local_id)`; the local ID is a nonempty string. Consumers
read the complete installed declaration snapshot, even for partial operations.
Unknown consumer names are errors. Known optional consumers that are not
installed do not consume declarations.

```python
from linktools.cntr import Flare, Integrations, Nginx
from linktools.cntr.ext import load_port_url


@cached_property
def integrations(self) -> "Integrations":
    return [
        Nginx.site(
            self.get_config_later("APP_DOMAIN"),
            proxy="http://app:8080",
            auth=None,
            auth_bypass=(r"^/public/",),
            waf_bypass=(),
            link=Flare.public("App", "apps", "Application"),
        ),
        Flare.container(
            "App direct", "apps",
            load_port_url(self, "APP_PORT", https=False),
        ),
    ]
```

`Nginx` and `Flare` provide the declaration factories. `Nginx.site(...)` returns
a `Nginx`; `Flare.public(...)`, `Flare.container(...)`, and `Flare.bookmark(...)`
return a `Flare`.
The `ext/` package owns these factories and public declaration types, also
re-exported from `linktools.cntr`. `Integration` declares only the public
`consumer` name. `Nginx.consumer` is `"nginx"`; `Flare.consumer` is `"flare"`.
New consumer-specific subclasses set their own `consumer` name. `Integrations`
is the Python 3.6-compatible alias
`Iterable[Integration]`: return a finite list, tuple, or iterator of declarations.
The former consumer-keyed mappings and nested named mappings are not accepted.

`Nginx.local_id` defaults to `"web"`, so omit it for a single/default site.
Give additional sites explicit IDs, for example `Nginx.site("api.example.com", local_id="api")`. IDs must be nonempty
strings and unique within their producer. Different producers can reuse the
same ID. Other integrations have no framework-level local ID and do not need
invented names.

The manager freezes each installed producer's declaration structure once per
command into a read-only producer mapping of declaration tuples, including
consuming a generator once. Mixed declaration order is preserved in the snapshot.
Iterables must be finite. If `integrations` is cached and its snapshot can be
rebuilt, return a reusable list/tuple or provide a fresh iterator for the rebuild;
do not reuse an exhausted generator.
It checks consumer names and the `Integration` type without resolving lazy
fields or URLs. Consumers validate their own declaration content. The nginx
container owns the command-local site snapshot in its public `sites` property.
`ResolvedSite.collect(manager)` in `ext` checks site types and nonempty, unique
local IDs before resolving any site fields. URL helpers, navigation and
authentication share `containers["nginx"].sites`; the manager has no nginx index.
The generic `iter_integrations` filters by the declaration's consumer and yields
`(producer, declaration)`. It returns only explicit declarations for an
installed consumer.

`Nginx.link` optionally attaches a `Flare` for navigation. Omitting
its URL lazily inherits the resolved site's URL. Explicit `None` or `""` disables
the link; an explicit URL stays unchanged. An omitted URL on a standalone link
has no site to inherit from and is skipped. The link's category is presentation
metadata, not an authentication policy; there is no `public` site boolean.
Sites without `link` create no navigation.

Standalone `Flare` values in the list supply independent navigation entries.
Use `Flare.public(name, icon, desc, url)` for application entries; their
descriptions are included in `apps.yml`. Use
`Flare.bookmark(name, icon, url, category="tool")` for bookmarks, which do not
need an application description. A custom string category creates a bookmark
group with that ID and title. The standard IDs `private`, `container`, and
`other` use their predefined titles and orders; omitting `category` uses `other`.
`Flare.container(name, icon, url, *, desc=None)` is shorthand for
`Flare.bookmark(name, icon, url, category="container", desc=desc)`, using the standard
`Internal` bookmark group. Its optional URL follows the same inheritance and
explicit `None`/empty behavior as other Flare declarations.
Both factories retain an optional `desc` on the declaration, defaulting to the
name when omitted or empty; bookmark output does not include descriptions.

Pass the internal category value created by
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
`Flare` factories for declarations. `NginxSite`, `FlareLink`, and `FlareCategory`
are not public exports. The former `exposes` property
and `self.expose_*` helpers are removed without aliases or fallbacks.
Links keep their category, name, icon, description and lazy URL; direct-port,
external and non-HTTP links do not need an nginx site. Flare preserves container
`order` (and snapshot order for ties). Within each producer, attached links follow
nginx declaration order, then independent links follow their declaration order.
App links keep traversal order across all app categories;
Category `order` applies to bookmark categories. Links within each bookmark
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

Import the URL factories from `linktools.cntr.ext` and pass the owning container:

```python
from linktools.cntr.ext import load_config_url, load_nginx_url, load_port_url

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
- `default_server=True` makes this site the native default server on its HTTP, HTTPS
  (when enabled), and WAF (when enabled) listeners, regardless of its hostname.
  Enabled defaults must not share a listening socket. If none is declared, nginx
  adds one built-in fallback site. `server_name="_"` alone no longer requests
  default behavior; migrate catch-all declarations to `default_server=True`
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
- Only a single literal hostname implies a public URL. Pattern/default sites
  need explicit `public_url`; navigation placeholders such as `{{port}}` remain literal
- `cert_domains` declares additional certificate names; `template_vars` owns business
  template variables

## Authelia OIDC callbacks

Declare callbacks independently of nginx sites:

```python
from linktools.cntr.ext import Authelia, load_nginx_url

Authelia.oidc(
    redirect_uris=(load_nginx_url(self, "web", "sso/callback"),),
    enabled=self.get_config_later("NGINX_AUTH_ENABLE"),
)
```

`Authelia.oidc` contributes URLs to the existing shared client; it never declares
a second client or changes client IDs, credentials, scopes, or authorization policy.
Use lazy URL/config references to defer evaluation. Disabled declarations do not
resolve callbacks. Empty URLs are omitted, including disabled optional services;
nonempty callbacks must be absolute and have no fragment or template placeholder.
Provider-specific URI rules remain subject to [native Authelia validation](https://www.authelia.com/configuration/identity-providers/openid-connect/clients/#redirect_uris).
Deduplication preserves exact strings and order,
including path, query, and trailing slash differences.

Move former `Nginx.site(..., oidc_redirects=...)` values to an `Authelia.oidc`
declaration. Replace empty/root-relative site callbacks with explicit lazy public
URLs, and keep existing enable conditions for optional services.

The built-in Authelia exposes a read-only `oidc_client` mapping with `client_id`,
`client_name`, `client_secret`, `issuer_url`, `authorization_url`, `token_url`,
`userinfo_url`, `user_identifier`, and tuple `scopes`/`redirect_uris`. Consumers
must not mutate or restore historical derived ACL/OIDC settings.

## Native nginx templates

A custom template owns the server's business directives and locations, not its
listeners, TLS, authentication endpoint or WAF forwarding. Context contains
`site`, `container`, `nginx`, `config`, and `template_vars`, plus `route_auth=True` only
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

### Nginx and SafeLine networks

SafeLine owns the `safeline-ce` bridge and its existing `SAFELINE_SUBNET_PREFIX`
(default `172.22.242`). All SafeLine services, including Tengine and management,
join only this bridge. When SafeLine is enabled, Nginx also joins it at `.253`,
while retaining its own `nginx` bridge for application backends. Tengine keeps
its fixed `.254` address. Nginx only references SafeLine's network; it does not
own or redefine its IPAM. Dependency-only SafeLine installations use the same
attachment rules as direct installations.

The SafeLine origin contract remains `http://nginx:<NGINX_WAF_PORT>` (default
`http://nginx:8000`). Compose supplies the `nginx` service DNS name on the shared
bridge; `nginx-origin` is an optional additional alias. The fixed `.253`/`.254`
addresses prevent either hop from retaining an obsolete peer IP after container
recreation. A normal restart usually retains its endpoint; recreation replaces
it. Neither service borrows the other's network namespace. Nginx's Unix-socket
healthcheck does not wait for Tengine; only Tengine's startup dependency points
to healthy Nginx.

Public requests enter Nginx's HTTP/HTTPS listeners and WAF-protected requests
are forwarded to Tengine at `.254:<NGINX_WAF_PORT>`. SafeLine returns them to
Nginx's separate HTTP origin listener at `nginx:<NGINX_WAF_PORT>`. That listener
runs the application/authentication routes without forwarding into WAF again.
SafeLine must preserve the original Host and `X-Proxy-Original-*` metadata;
Nginx validates those values and accepts them only from the exact Tengine peer,
not the entire subnet. Pointing the origin at the public HTTP/HTTPS listener
instead would create a forwarding loop. The WAF port is not published to the
host. `SAFELINE_PORT=0` disables the optional management host port; its default
remains `9200` mapped to `1443`.

No new subnet setting or Nginx network recreation is required. Existing valid
`http://nginx:<NGINX_WAF_PORT>` origins remain valid and are not rewritten.
Changing `SAFELINE_SUBNET_PREFIX` separately remains a network migration, not a
partial service update. Check it against host/VPN/other Docker routes before
changing it. Scoped SafeLine removal does not remove a bridge still used by
Nginx; after uninstalling SafeLine, reapplying Nginx without WAF removes its old
attachment. The tool does not automatically tear down shared networks.

`dependencies` defines required installed dependencies and startup order
for an explicitly selected container. A different running service selected only
for configuration reconciliation follows its own Compose service dependencies,
without automatically starting the other dependencies of its owning container.
An enabled nginx site also needs its installed nginx runtime provider when the
producer is explicitly selected. Flare navigation is optional and never creates
a startup dependency. Explicitly selecting Flare still starts it.

## Operation lifecycle

Container declarations remain independent of execution. Prepared files, checks,
service recreation, recovery and ACME handling are specified in
[the lifecycle contract](lifecycle.md). There is no separate generated-config
or bootstrap interface on BaseContainer.

## Module path migration

Import extension declarations and URL factories from `linktools.cntr.ext`.
Downstream code using the former `linktools.cntr.integration` package must update
its imports; the former package is not retained as a compatibility alias.
The `integrations` declaration property and existing root-level declaration exports
keep their names and behavior.
