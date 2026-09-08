#!/usr/bin/env bash
# Resolve with the signed snapshot indexes, then install only verified cache files.
set -euo pipefail

apt-get -o Acquire::ForceHash=SHA256 --yes --no-install-recommends \
  --print-uris --download-only install "$@" > /tmp/apt-download-plan.txt
sed -n -E "s/^'([^']+)' ([^ ]+) ([0-9]+) (SHA256:[a-f0-9]{64})$/\1\t\2\t\3\t\4/p" \
  /tmp/apt-download-plan.txt > /opt/jax-environment/apt-downloads.tsv

export APT_SNAPSHOT APT_MIRRORS
xargs -r -d '\n' -n 1 -P 4 bash /opt/jax-environment/fetch-apt-package.sh \
  < /opt/jax-environment/apt-downloads.tsv
apt-get -o Acquire::ForceHash=SHA256 --yes --no-install-recommends --no-download install "$@"
