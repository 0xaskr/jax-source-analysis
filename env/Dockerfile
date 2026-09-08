ARG BASE_IMAGE=ubuntu@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517
FROM ${BASE_IMAGE}

ARG APT_SNAPSHOT
ARG APT_PACKAGES
ARG APT_MIRRORS
ARG RECIPE_SHA256
LABEL org.jax-source-analysis.environment=${RECIPE_SHA256}

# This bundle comes only from the hash-verified Ubuntu CA package. The minimal
# base image has no CA bundle with which to authenticate the snapshot service.
COPY ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
COPY install-packages.sh fetch-apt-package.sh /opt/jax-environment/
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/sources.list /etc/apt/sources.list.d/ubuntu.sources /etc/apt/apt.conf.d/docker-clean \
    && printf 'deb https://snapshot.ubuntu.com/ubuntu/%s noble main universe\ndeb https://snapshot.ubuntu.com/ubuntu/%s noble-updates main universe\ndeb https://snapshot.ubuntu.com/ubuntu/%s noble-security main universe\n' \
       "$APT_SNAPSHOT" "$APT_SNAPSHOT" "$APT_SNAPSHOT" > /etc/apt/sources.list \
    && apt-get -o APT::Update::Error-Mode=any -o Acquire::Retries=3 -o Acquire::https::Timeout=60 update \
    && DEBIAN_FRONTEND=noninteractive bash /opt/jax-environment/install-packages.sh $APT_PACKAGES

COPY uv /usr/local/bin/uv
COPY environment.lock.json verify-image.py /opt/jax-environment/
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN /usr/bin/python3.12 -B /opt/jax-environment/verify-image.py \
    && dpkg-query -W > /opt/jax-environment/installed-packages.txt

CMD ["/usr/bin/python3.12", "--version"]
