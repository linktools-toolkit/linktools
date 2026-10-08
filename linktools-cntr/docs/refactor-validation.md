# Integration validation and deployment gates

Date: 2026-10-08.

## Scope and compatibility

The paired cntr and homelab changes use one public declaration contract:
`Integration`, `Integrations`, `NginxSite`, `ExposeCategory`, and `ExposeLink` in
`linktools.cntr.integration`. Named mappings retain stable local IDs; finite
iterables are available to consumers that do not require names. Nginx requires
named sites. Flare accepts ordered unnamed links and site-attached navigation.
See [integrations.md](integrations.md) for the authoring API and lazy URL helpers.

The authoring surface remains `configs`, `dependencies`, and `integrations`.
The removed BaseContainer generation hooks, `config_sources`, exposure helpers,
and URL mixins are not compatibility APIs. Four internal builtin owners handle
configuration generation. Complete candidate comparison deliberately allows a
partial `up` or `restart` to apply pending changes to other running services;
unrelated stopped services stay stopped unless required by runtime dependencies.

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

- Declaration snapshots: pure types, named and unnamed collections, one-time
  finite-iterator consumption, immutable structure, insertion order, disabled
  consumers, and lazy values that are not read while declarations are collected
- URL factories: delayed configuration/host/site reads, immediate local-ID
  validation, missing and disabled values, per-producer site identities, and
  unchanged path/query composition
- Navigation: original 65 homelab and seven builtin links, site-attached before
  standalone links, standard category order, and the Authelia `/auth-admin` link
- Native templates: explicit namespaces, no removed snippet references, runtime
  Docker DNS, preserved prefix/capture/query routing, and complete header macros
- Lifecycle: restoration after failed nginx bootstrap, matching applied Compose
  snapshots, certificate replacement despite unchanged config IDs, aligned ACME
  cron configuration, and explicit/implicit Compose dependency ordering
- Runtime arguments: literal-dollar decoding only at the raw Docker boundary,
  retaining correctly escaped persisted Compose files for later rollback
- Paired commands: MCP Playwright and MCP Push `show` handlers, including Push's
  optional channel argument

## Remaining acceptance gates

| Area | Remaining target-environment validation |
| --- | --- |
| Native nginx | Full migrated homelab route set, HTTP/2 gRPC, SSE/WebSocket, and unusual custom headers |
| Authentication | Real Authelia sessions, OIDC flows, and alternate-authentication provenance assurance |
| SafeLine | Actual target images, Docker networks, request metadata, and forwarding behavior |
| Certificates | Real ACME issuance, renewal/cron reload, provider credentials, and persisted account reuse |
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
