# Container integrations

`integrations` declares values for installed consumers. `NginxSite` identifies a
site by `(producer.name, local_id)`; the local ID is a nonempty string. Consumers
read the complete installed declaration snapshot, even for partial operations.
Unknown consumer names are errors. Known optional consumers that are not
installed do not consume declarations.

```python
@cached_property
def integrations(self) -> "dict[str, dict[str, Any]]":
    return {"nginx": {"web": NginxSite(
        server_name=self.get_config_later("APP_DOMAIN"),
        proxy="http://app:8080",
        auth=None,
        auth_bypass=(r"^/public/",),
        waf_bypass=(),
    )}, "flare": {"web": self.expose_public(
        "App", "apps", "Application", self.load_nginx_url("web"),
    )}}
```

`integrations["flare"][local_id]` supplies an `ExposeLink` for navigation.
The former `exposes` property is removed without an alias or fallback.
Links keep their category, name, icon, description and lazy URL; direct-port,
external and non-HTTP links do not need an nginx site. Flare preserves container
`order` and each producer’s declaration insertion order, skips empty URLs and
rejects conflicting descriptions for the same category. `load_nginx_url("web", "ui")` lazily reads
that site's URL; it never registers a hook or writes a configuration. A site
without navigation still generates a proxy. The former registration arguments,
`write_nginx_conf`, `load_exist_nginx_url`, and `append_ssl_domains` are removed.
Migrate callers and templates together; there is no compatibility wrapper.

## Site values

- `server_name=""` disables the site and unrelated secret/config resolution
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
listeners, TLS, authentication endpoint or WAF forwarding. Context is exactly
`site`, `container`, `nginx`, `config`, and `vars`. Read environment values with
`config.get("KEY")`. Undefined fields fail rendering.

Jinja names start with `local/` (the entry template's directory tree) or
`nginx/` (builtin templates). Business templates render once. Native nginx
`include` loads generated output and is distinct from a Jinja include. Data is
not rendered again; use Jinja comments to disable logic and raw blocks to emit
another template language.

```jinja
{% from "nginx/headers.j2" import proxy_headers with context %}
location / {
    {{ proxy_headers({"Host": "$host"}) }}
    set $upstream "http://app:8080";
    proxy_pass $upstream;
}
```

Header macros emit a complete same-level set. Overrides are single complex
values, not directives or prequoted strings. Names are case-insensitive;
framework identity/internal metadata headers and auth-header conflicts cannot
be overridden. Custom `auth_request off`, `return`, rewrite and native location
selection retain nginx semantics. An early `return` is not protected by the
access phase. Preserve URI replacement and captures when migrating static
upstreams to Docker runtime DNS.

## Selection, publication and migration

`dependencies` defines required installed dependencies and startup order.
An enabled nginx site also needs its installed nginx runtime provider. Flare
navigation is optional and never creates a startup dependency. Explicitly
selecting Flare still starts it.

Every `up` or `restart` renders the full installed candidate configuration and
compares it with the last applied, fully resolved Compose service models and
generated trees. Resolved snapshots preserve environment values from `.env`,
`env_file` and Compose interpolation for comparison and rollback. Shared top-level
network/volume changes conservatively invalidate running service models.
Explicit targets and their runtime dependencies are ensured running. Other
services are updated only when their configuration changed and they are already
running; stopped sibling services are not started by configuration reconciliation.
A partial command can therefore apply pending changes to other running services.
When no resolved snapshot exists yet, a running service is reconciled once to
establish it; rollback uses the previous saved Compose file where available.
Historical external environment-file contents cannot be recovered retroactively.
`restart app` stops only explicit targets, after all candidate validation passes.
Preparation covers running services and their possible runtime dependencies;
application still uses the final changed-service selection. Snapshots are captured
after startup hooks, so hook-prepared environment files are included.
The plan reports the full reconciliation scope and defers runtime-dependent
update decisions until execution.

Container authors do not declare `config_sources` or an integration startup
policy. OIDC/template cross-container reads are captured by complete candidate
rendering. Aggregate consumers read the complete installed snapshot, including
removal of a producer's final declaration. No historical dependency graph or
runtime configuration-read tracking is needed.

The only new container authoring entry point is `integrations`. Generation
paths and preparation, rendering, native validation, application and rollback
belong to the four bundled consumer implementations. nginx-specific template
rendering also belongs to its consumer; none of those operations is a
`BaseContainer` extension hook. This navigation and synchronization addendum
supersedes the earlier Site protocol's separate `exposes` and `config_sources`
properties. Migrate both repositories together; there is no fallback alias.

Generated configuration uses a stable mounted parent, immutable generation
folders and an atomic `current` link. Candidates are rendered and validated
before restart stops a target. nginx loads bootstrap health/rejection config
when starting without a serving process; authentication/WAF providers become
ready before the complete nginx config is activated. nginx health returns the
loaded generation ID, and failed application restores the previous generation.
Cross-service state is not an atomic transaction; failures remain command errors.

Deployment migration must be coordinated with all external repository callers:

1. Back up existing generated config and record the matching repository commits,
   image IDs and SafeLine origin target. Preserve settings, keys and databases
2. Migrate external Python declarations, native templates and OIDC consumers
3. Recreate nginx for the stable generated-parent mount change
4. Point this integration's SafeLine origin to
   `http://nginx-origin:<NGINX_WAF_PORT>` and preserve all five `X-Cntr-*` headers
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
