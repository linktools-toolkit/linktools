# syntax=docker/dockerfile:1
FROM nginx:{{ NGINX_TAG }}

# {% if NGINX_HTTPS_ENABLE %}
RUN apk --no-cache add openssl socat coreutils

RUN curl https://get.acme.sh | sh -s -- --home /opt/acme --config-home /root/.acme.sh --nocron && \
    ln -s /opt/acme/acme.sh /usr/bin/acme.sh

COPY nginx-init-certificates nginx-reload /usr/local/bin/
RUN chmod +x /usr/local/bin/nginx-init-certificates /usr/local/bin/nginx-reload && \
    mkdir -p /etc/certs && \
    printf '#!/bin/sh\nset -e\n/usr/local/bin/nginx-init-certificates\ncrond\n' > /docker-entrypoint.d/99-start-crond.sh && \
    chmod +x /docker-entrypoint.d/99-start-crond.sh && \
    printf '17 2 * * * /opt/acme/acme.sh --cron --home /opt/acme --config-home /root/.acme.sh > /proc/1/fd/1 2> /proc/1/fd/2\n' > /etc/crontabs/root

RUN --mount=type=secret,id=nginx-acme,required=true \
    --mount=type=tmpfs,target=/root/.acme.sh \
    . /run/secrets/nginx-acme && \
    acme.sh --home /opt/acme --config-home /root/.acme.sh --server {{ container.shell_quote(ACME_SERVER) }} \
        --issue {{ manager.integration_consumers['nginx'].acme_ssl_domains_args }} --dns {{ container.shell_quote(ACME_DNS_API) }}{% if ACME_ACCOUNT_EMAIL %} --accountemail {{ container.shell_quote(ACME_ACCOUNT_EMAIL) }}{% endif %} && \
    acme.sh --home /opt/acme --config-home /root/.acme.sh --server {{ container.shell_quote(ACME_SERVER) }} \
        --install-cert {{ manager.integration_consumers['nginx'].acme_ssl_domains_args }} {{ manager.integration_consumers['nginx'].acme_ssl_certificate_args }} \
        --reloadcmd /usr/local/bin/nginx-reload && \
    rm -f /root/.acme.sh/account.conf && \
    for config in /root/.acme.sh/*/*.conf; do \
        [ -f "$config" ] || continue; \
        sed -n '/^Le_[A-Za-z0-9_]*=/p' "$config" > "$config.clean" && \
        mv "$config.clean" "$config" || exit 1; \
    done && \
    mkdir -p /opt/nginx-initial/certs /opt/nginx-initial/acme && \
    cp -a /etc/certs/. /opt/nginx-initial/certs/ && \
    cp -a /root/.acme.sh/. /opt/nginx-initial/acme/
# {% endif %}
