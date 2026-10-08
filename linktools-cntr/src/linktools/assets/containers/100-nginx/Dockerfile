# syntax=docker/dockerfile:1.7
# {% if NGINX_HTTPS_ENABLE %}
FROM nginx:{{ NGINX_TAG }} AS acme-build

RUN apk --no-cache add openssl socat coreutils

RUN curl https://get.acme.sh | sh -s -- --home /opt/acme --config-home /root/.acme.sh --nocron

# {% for key, field in container.extend_configs.items() %}
ENV {{ key }}={{ container.acme_dns_environment_value(field) }}
# {% endfor %}

RUN --mount=type=secret,id=cntr_acme_account \
    mkdir -p /root/.acme.sh /etc/certs && \
    if [ -s /run/secrets/cntr_acme_account ]; then \
        tar -xf /run/secrets/cntr_acme_account -C /root/.acme.sh; \
    fi && \
    /opt/acme/acme.sh --home /opt/acme --config-home /root/.acme.sh --server {{ container.shell_quote(ACME_SERVER) }} \
        --issue {{ container.acme_ssl_domains_args }} --dns {{ container.shell_quote(ACME_DNS_API) }}{% if ACME_ACCOUNT_EMAIL %} --accountemail {{ container.shell_quote(ACME_ACCOUNT_EMAIL) }}{% endif %} && \
    /opt/acme/acme.sh --home /opt/acme --config-home /root/.acme.sh --server {{ container.shell_quote(ACME_SERVER) }} \
        --install-cert {{ container.acme_ssl_domains_args }} {{ container.acme_ssl_certificate_args }} \
        --reloadcmd ':' && \
    mkdir -p /opt/nginx-initial/certs /opt/nginx-initial/acme && \
    cp -a /etc/certs/. /opt/nginx-initial/certs/ && \
    cp -a /root/.acme.sh/. /opt/nginx-initial/acme/
# {% endif %}

FROM nginx:{{ NGINX_TAG }}

# {% if NGINX_HTTPS_ENABLE %}
RUN apk --no-cache add openssl socat coreutils util-linux
COPY --from=acme-build /opt/acme /opt/acme
COPY --from=acme-build /opt/nginx-initial /opt/nginx-initial
COPY nginx-certificates nginx-reload /usr/local/bin/
RUN chmod +x /usr/local/bin/nginx-certificates /usr/local/bin/nginx-reload && \
    chmod -R go-rwx /opt/nginx-initial && \
    ln -s /opt/acme/acme.sh /usr/bin/acme.sh && \
    printf '#!/bin/sh\nset -e\ncrond\n' > /docker-entrypoint.d/99-start-crond.sh && \
    chmod +x /docker-entrypoint.d/99-start-crond.sh && \
    printf '17 2 * * * /usr/local/bin/nginx-certificates renew > /proc/1/fd/1 2> /proc/1/fd/2\n' > /etc/crontabs/root
# {% endif %}
