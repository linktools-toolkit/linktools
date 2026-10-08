# Refactor validation and deployment gates

Date: 2026-10-08.

## Code baselines and scope

- Implementation base: `linktools-toolkit/linktools`,
  `refactor/cntr-integrations@023eb9cc3b367049e0bc896c1001b83a99040c4d`
- Read-only compatibility reference: `linktools-toolkit/linktools-homelab`,
  `refactor/cntr-integrations@15e3dfe070c3a26cfd6a75b615cda946c984dd0c`
- This change implements the cntr side and the minimal core read-only resolution
  API required by dry-run. It does not migrate or publish the homelab repository
- No version bump, remote push, live service deployment, or real credential
  rotation was performed

## Reproducible local checks

Run the normal gate with the repository's installed dependencies:

```sh
python manage.py check linktools linktools-cntr
python manage.py build linktools linktools-cntr
python manage.py verify linktools linktools-cntr
```

The optional real-nginx request test uses an explicitly provided nginx binary:

```sh
CNTR_TEST_NGINX=/path/to/nginx python manage.py check linktools-cntr
```

This environment compiled official nginx **1.28.0**, with HTTP SSL, HTTP/2,
real-IP and auth-request modules. It disallows Unix-domain listeners. Local
request tests therefore set `CNTR_NGINX_TEST_TCP_HEALTH=1`, an explicit test-only
transformation from the production Unix health listener to loopback TCP.
Production generated configuration continues to use the Unix socket.

The normal native request regression covers HTTP/TLS, the four auth/WAF bypass
combinations with counted mock providers, binary POST and encoded URI/query,
request metadata, conventional client/service credentials, numeric captures,
401/403/provider errors, public fallback, rejected origin sources/metadata,
SSE delivery, and a real WebSocket handshake plus binary-frame echo. It does
not certify real SafeLine behavior. An additional alternate-authentication
provenance follow-up test was not completed; its code change remains a
specifically unverified boundary.

Focused regressions cover lazy declarations and disabled sites; strict URL and
OIDC rules; literal data and Jinja namespaces; custom Flare categories; immutable
candidate reuse; publish/apply/rollback failure; partial target/provider/config
source ordering; per-service Compose drift; read-only planning and missing
secret handling; persistent data preservation; Flare migration rollback and
service-group permissions; and stopped legacy nginx certificate preservation.

## Spec disposition

| Acceptance area | Local evidence | Remaining gate |
| --- | --- | --- |
| Declarations and ownership | Shared lazy snapshot, pure URL lookup, removed old Python registration APIs, focused tests | Complete homelab migration |
| Switches, URL and OIDC | Resolution/provider/HTTPS tests, immutable rebuilt callbacks | Real browser session and OIDC flows |
| Jinja and headers | Strict namespaces, literal-dollar data, same-level macros, single business render | Unusual custom native header configurations |
| Request paths | Real nginx normal request matrix, SSE and WebSocket | HTTP/2 gRPC and full homelab route set |
| Authorization boundaries | Normal denial/bypass/fallback tests | Additional alternate-authentication provenance follow-up |
| SafeLine | Generated addresses and metadata contract, mock routing | Target image IDs, actual Docker network and SafeLine forwarding |
| Derived data | ACL/OIDC reconstruction and candidate/password/database preservation tests | Real LDAP/password interoperability; changing a config value does not reset an existing LDAP administrator password |
| Lifecycle | Selection, ordering, isolated command model, cold bootstrap and readiness mocks | Actual HTTP/WAF and HTTPS/auth cold starts, DNS/IP changes |
| Failure and rollback | Immutable trees, atomic symlink, saved applied Compose, failure injection | Real bind mounts, Unix socket permissions, reload/rollback acknowledgements and concurrent Docker operations |
| Delivery | Repository tests/static checks and local packaging checks | All target-environment gates before deployment acceptance |

LLDAP has no standalone configuration-validation subcommand in its upstream
CLI. Its fixed builtin TOML and derived password are staged without touching
`/data`, then service health is required after application. Native nginx and
Authelia candidate validators use isolated target-image containers in the
implementation; those Docker invocations were not run in this environment.

Legacy nginx migration preserves certificate/ACME/config backups without
replacing an earlier backup, including existing stopped containers. Real Docker
copy/mount behavior and ACME issuance/renewal still require deployment testing.

## Known homelab migration blockers

At the reference commit, enabled custom sites in these files still include
removed `/etc/nginx/conf.d/snippets/*` native files:

- `2xx-homelab/221-fnos/nginx.conf`
- `3xx-proxy/380-sublink/nginx.conf`
- `5xx-ai/500-vscode/proxy.conf` (wildcard proxy site)
- `5xx-ai/501-aionui/nginx.conf`
- `5xx-ai/510-hermes-agent/nginx.conf`
- `5xx-ai/520-multica-server/nginx.conf`
- `5xx-ai/550-mcp-playwright/nginx.conf`
- `5xx-ai/551-mcp-push/nginx.conf`

The MCP Playwright and MCP Push show commands also still call removed
`load_exist_nginx_url`. Nextcloud, qBittorrent and pypiserver retain static Docker
upstreams; those templates and Xray need the new metadata/header macros.
Xray's configured service/path normalization is not yet shared between routes
and bypass declarations. GitLab and LiteLLM already use `oidc_client`, but lack
`config_sources=("authelia",)` for partial configuration propagation.

Because candidates include all installed enabled sites, an incompatible site
also blocks a partial nginx update. Pair the repository migration explicitly;
`>=0.10.0` does not distinguish these unversioned protocol changes. Existing
homelab documentation still describes removed behavior and needs updating.

**Local tests are not deployment acceptance.** Do not publish these configurations
to an existing homelab until its callers are migrated and the real Docker,
SafeLine, Authelia/OIDC, certificate and Unix health gates above are verified.
