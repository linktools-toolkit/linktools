# syntax=docker/dockerfile:1.7
# {% if NGINX_HTTPS_ENABLE %}
FROM alpine:3.23.6 AS acme

RUN apk --no-cache add ca-certificates curl openssl socat coreutils
RUN curl -fsSL https://github.com/acmesh-official/acme.sh/archive/master.tar.gz -o /tmp/acme.tar.gz && \
    tar -xzf /tmp/acme.tar.gz -C /tmp && \
    cd /tmp/acme.sh-master && \
    sh acme.sh --install --home /opt/acme --config-home /root/.acme.sh --nocron --noprofile && \
    test -x /opt/acme/acme.sh && \
    cd / && rm -rf /tmp/acme.sh-master /tmp/acme.tar.gz
COPY nginx-certificates nginx-acme /usr/local/bin/
RUN printf '%s\n' {% for api in container.dnsapi.values() %}{% for key in api.get("env", {}) %}{{ container.shell_quote(key) }} {% endfor %}{% endfor %}| sort -u > /opt/acme/dns-keys
RUN chmod +x /usr/local/bin/nginx-certificates /usr/local/bin/nginx-acme && \
    ln -s /opt/acme/acme.sh /usr/bin/acme.sh

ARG CNTR_BUILD_REVISION
RUN --mount=type=secret,id=cntr_acme_account \
    --mount=type=secret,id=cntr_acme_dns,target=/run/acme-secrets/dns.env,required=true \
    --mount=type=tmpfs,target=/root/.acme.sh \
    test -n "$CNTR_BUILD_REVISION" && mkdir -p /root/.acme.sh /etc/certs && \
    if [ -s /run/secrets/cntr_acme_account ]; then \
        tar -xf /run/secrets/cntr_acme_account -C /root/.acme.sh; \
    fi && \
    /usr/local/bin/nginx-acme /root/.acme.sh --server {{ container.shell_quote(ACME_SERVER) }} \
        --issue --force {{ container.acme_ssl_domains_args }} --dns {{ container.shell_quote(ACME_DNS_API) }}{% if ACME_ACCOUNT_EMAIL %} --accountemail {{ container.shell_quote(ACME_ACCOUNT_EMAIL) }}{% endif %} && \
    /usr/local/bin/nginx-acme /root/.acme.sh --server {{ container.shell_quote(ACME_SERVER) }} \
        --install-cert {{ container.acme_ssl_domains_args }} {{ container.acme_ssl_certificate_args }} \
        --reloadcmd ':' && \
    mkdir -p /opt/nginx-initial/certs /opt/nginx-initial/acme && \
    cp -a /etc/certs/. /opt/nginx-initial/certs/ && \
    cp -a /root/.acme.sh/. /opt/nginx-initial/acme/ && \
    printf '%s\n' {{ container.shell_quote(container.cert_image_revision) }} > /opt/nginx-initial/revision && \
    printf '%s\n' {{ container.shell_quote(NGINX_ROOT_DOMAIN) }} > /opt/nginx-initial/primary && \
    printf '%s\n' {% for domain in container.acme_ssl_domains %}{{ container.shell_quote(domain) }} {% endfor %}> /opt/nginx-initial/domains && \
    chmod -R go-rwx /opt/nginx-initial

# {% endif %}

FROM nginx:{{ NGINX_TAG }} AS nginx

# {% if NGINX_HTTPS_ENABLE %}
RUN apk --no-cache add ca-certificates curl openssl socat coreutils util-linux
COPY --from=acme /opt/acme /opt/acme
COPY --from=acme /opt/nginx-initial /opt/nginx-initial
COPY --from=acme /usr/local/bin/nginx-acme /usr/local/bin/nginx-certificates /usr/local/bin/
RUN ln -s /opt/acme/acme.sh /usr/bin/acme.sh
LABEL io.linktools.nginx.certificate-revision="{{ container.cert_image_revision }}"

RUN printf '#!/bin/sh\nset -e\n/usr/local/bin/nginx-certificates install\ncrond\n' > /docker-entrypoint.d/99-certificates.sh && \
    chmod +x /docker-entrypoint.d/99-certificates.sh && \
    printf '17 2 * * * /usr/local/bin/nginx-certificates renew > /proc/1/fd/1 2> /proc/1/fd/2\n' > /etc/crontabs/root

ARG CNTR_BUILD_REVISION
RUN --mount=type=secret,id=cntr_acme_dns,target=/run/acme-secrets/dns.env,required=true \
    set -eu; \
    test -n "$CNTR_BUILD_REVISION"; \
    /usr/local/bin/nginx-acme /opt/nginx-initial/acme --upgrade --force; \
    /usr/local/bin/nginx-acme /opt/nginx-initial/acme \
        --install-cert --domain {{ container.shell_quote(NGINX_ROOT_DOMAIN) }} \
        --cert-file /opt/nginx-initial/certs/{{ container.shell_quote(NGINX_ROOT_DOMAIN) }}_cert.pem \
        --key-file /opt/nginx-initial/certs/{{ container.shell_quote(NGINX_ROOT_DOMAIN) }}_key.pem \
        --fullchain-file /opt/nginx-initial/certs/{{ container.shell_quote(NGINX_ROOT_DOMAIN) }}_fullchain.pem \
        --reloadcmd ':'; \
    status=0; \
    /usr/local/bin/nginx-acme /opt/nginx-initial/acme \
        --renew --domain {{ container.shell_quote(NGINX_ROOT_DOMAIN) }} || status=$?; \
    test "$status" -eq 0 || test "$status" -eq 2; \
    chmod -R go-rwx /opt/nginx-initial
# {% endif %}
