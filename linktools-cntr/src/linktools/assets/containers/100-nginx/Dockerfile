# syntax=docker/dockerfile:1.7
ARG NGINX_TAG=stable-alpine
FROM nginx:${NGINX_TAG}

# {% if NGINX_HTTPS_ENABLE %}
RUN apk --no-cache add openssl socat coreutils util-linux
RUN curl -fsSL https://github.com/acmesh-official/acme.sh/archive/master.tar.gz -o /tmp/acme.tar.gz && \
    tar -xzf /tmp/acme.tar.gz -C /tmp && \
    cd /tmp/acme.sh-master && \
    sh acme.sh --install --home /opt/acme --config-home /root/.acme.sh --nocron --noprofile && \
    test -x /opt/acme/acme.sh && \
    cd / && rm -rf /tmp/acme.sh-master /tmp/acme.tar.gz
COPY <<'NGINX_CERTIFICATES_SCRIPT' /usr/local/bin/nginx-certificates
{% include 'nginx-certificates' %}
NGINX_CERTIFICATES_SCRIPT
COPY <<'NGINX_ACME_SCRIPT' /usr/local/bin/nginx-acme
{% include 'nginx-acme' %}
NGINX_ACME_SCRIPT
RUN printf '%s\n' {% for api in container.dnsapi.values() %}{% for key in api.get("env", {}) %}{{ container.shell_quote(key) }} {% endfor %}{% endfor %}| sort -u > /opt/acme/dns-keys
RUN chmod +x /usr/local/bin/nginx-certificates /usr/local/bin/nginx-acme && \
    ln -s /opt/acme/acme.sh /usr/bin/acme.sh

ARG NGINX_ROOT_DOMAIN
ARG NGINX_CERT_DOMAINS
ARG ACME_SERVER
ARG ACME_DNS_API
ARG ACME_ACCOUNT_EMAIL
ARG ACME_DNS_CREDENTIALS_DIGEST
RUN --mount=type=secret,id=cntr_acme_account \
    --mount=type=secret,id=cntr_acme_dns,target=/run/acme-secrets/dns.env,required=true \
    --mount=type=tmpfs,target=/root/.acme.sh <<'BUILD_CERTIFICATES'
set -eu
test -n "$ACME_DNS_CREDENTIALS_DIGEST"
mkdir -p /root/.acme.sh /etc/certs
if [ -s /run/secrets/cntr_acme_account ]; then
    tar -xf /run/secrets/cntr_acme_account -C /root/.acme.sh
fi
set --
while IFS= read -r domain || test -n "$domain"; do
    test -n "$domain" && set -- "$@" --domain "$domain"
done <<DOMAINS
$NGINX_CERT_DOMAINS
DOMAINS
if test -n "$ACME_ACCOUNT_EMAIL"; then
    set -- "$@" --accountemail "$ACME_ACCOUNT_EMAIL"
fi
/usr/local/bin/nginx-acme /root/.acme.sh --server "$ACME_SERVER" \
    --issue --force "$@" --dns "$ACME_DNS_API"
/usr/local/bin/nginx-acme /root/.acme.sh --server "$ACME_SERVER" \
    --install-cert -d "$NGINX_ROOT_DOMAIN" \
    --cert-file "/etc/certs/${NGINX_ROOT_DOMAIN}_cert.pem" \
    --key-file "/etc/certs/${NGINX_ROOT_DOMAIN}_key.pem" \
    --fullchain-file "/etc/certs/${NGINX_ROOT_DOMAIN}_fullchain.pem" --reloadcmd ':'
mkdir -p /opt/nginx-initial/certs /opt/nginx-initial/acme
cp -a /etc/certs/. /opt/nginx-initial/certs/
cp -a /root/.acme.sh/. /opt/nginx-initial/acme/
printf '%s\n' "$NGINX_ROOT_DOMAIN" > /opt/nginx-initial/primary
printf '%s\n' "$NGINX_CERT_DOMAINS" > /opt/nginx-initial/domains
printf '%s\n' "$ACME_SERVER" "$ACME_DNS_API" "$ACME_ACCOUNT_EMAIL" > /opt/nginx-initial/settings
chmod -R go-rwx /opt/nginx-initial
BUILD_CERTIFICATES

RUN printf '#!/bin/sh\nset -e\n/usr/local/bin/nginx-certificates install\ncrond\n' > /docker-entrypoint.d/99-certificates.sh && \
    chmod +x /docker-entrypoint.d/99-certificates.sh && \
    printf '17 2 * * * /usr/local/bin/nginx-certificates renew > /proc/1/fd/1 2> /proc/1/fd/2\n' > /etc/crontabs/root
# {% endif %}
