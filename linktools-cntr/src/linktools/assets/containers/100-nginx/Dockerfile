# syntax=docker/dockerfile:1.7
FROM nginx:{{ NGINX_TAG }}

# {% if NGINX_HTTPS_ENABLE %}
RUN apk --no-cache add openssl socat coreutils util-linux
RUN curl https://get.acme.sh | sh -s -- --home /opt/acme --config-home /root/.acme.sh --nocron
COPY nginx-certificates nginx-acme /usr/local/bin/
LABEL io.linktools.nginx.certificate-revision="{{ container.cert_image_revision }}"
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

RUN printf '#!/bin/sh\nset -e\n/usr/local/bin/nginx-certificates install\ncrond\n' > /docker-entrypoint.d/99-certificates.sh && \
    chmod +x /docker-entrypoint.d/99-certificates.sh && \
    printf '17 2 * * * /usr/local/bin/nginx-certificates renew > /proc/1/fd/1 2> /proc/1/fd/2\n' > /etc/crontabs/root
# {% endif %}
