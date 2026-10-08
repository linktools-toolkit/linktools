FROM nginx:{{ NGINX_TAG }}

# {% if NGINX_HTTPS_ENABLE %}
RUN apk --no-cache add openssl socat coreutils

RUN curl https://get.acme.sh | sh -s email=acme@google.com --home /opt/acme && \
    ln -s /opt/acme/acme.sh /usr/bin/acme.sh && \
    acme.sh --set-default-ca --server letsencrypt

RUN mkdir -p /etc/certs && \
    echo "#!/bin/sh" >> /docker-entrypoint.d/99-start-crond.sh && \
    echo "crond" >> /docker-entrypoint.d/99-start-crond.sh && \
    chmod +x /docker-entrypoint.d/99-start-crond.sh

# {% endif %}
