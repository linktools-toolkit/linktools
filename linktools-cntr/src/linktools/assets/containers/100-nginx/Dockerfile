FROM nginx:{{ NGINX_TAG }}

# {% if NGINX_HTTPS_ENABLE %}
RUN apk --no-cache add openssl socat coreutils util-linux

RUN curl https://get.acme.sh | sh -s -- --home /opt/acme --config-home /root/.acme.sh --nocron && \
    ln -s /opt/acme/acme.sh /usr/bin/acme.sh

COPY nginx-certificates nginx-reload /usr/local/bin/
RUN chmod +x /usr/local/bin/nginx-certificates /usr/local/bin/nginx-reload && \
    printf '#!/bin/sh\nset -e\ncrond\n' > /docker-entrypoint.d/99-start-crond.sh && \
    chmod +x /docker-entrypoint.d/99-start-crond.sh && \
    printf '17 2 * * * /usr/local/bin/nginx-certificates renew > /proc/1/fd/1 2> /proc/1/fd/2\n' > /etc/crontabs/root
# {% endif %}
