#!/usr/bin/env bash
# Mirrors may supply bytes, but only the snapshot's SHA-256 authorizes installation.
set -euo pipefail
IFS=$'\t' read -r original filename size identity <<< "$1"
prefix="https://snapshot.ubuntu.com/ubuntu/${APT_SNAPSHOT}/"
[[ "$original" == "${prefix}pool/"* ]]
[[ "$filename" =~ ^[A-Za-z0-9.+:%~_-]+\.deb$ ]]
[[ "$size" =~ ^[0-9]+$ && "$identity" =~ ^SHA256:[a-f0-9]{64}$ ]]

cache=/var/cache/apt/archives
destination="$cache/$filename"
[[ ! -L "$destination" ]]
digest="${identity#SHA256:}"
valid_file() {
  [[ -f "$1" && "$(stat -c %s -- "$1")" == "$size" ]] \
    && printf '%s  %s\n' "$digest" "$1" | sha256sum --check --status
}
if valid_file "$destination"; then
  exit 0
fi
mkdir -p "$cache/partial"
temporary=$(mktemp "$cache/partial/environment-package.XXXXXX")
trap 'rm -f -- "$temporary"' EXIT

relative="${original#"$prefix"}"
read -r -a mirrors <<< "$APT_MIRRORS"
candidates=()
for mirror in "${mirrors[@]}"; do
  [[ "$mirror" == https://archive.ubuntu.com/ubuntu || "$mirror" == https://security.ubuntu.com/ubuntu ]]
  candidates+=("$mirror/$relative")
done
candidates+=("$original")
for candidate in "${candidates[@]}"; do
  if /usr/lib/apt/apt-helper -o Acquire::Retries=1 -o Acquire::https::Timeout=45 \
      download-file "$candidate" "$temporary" "$identity" && valid_file "$temporary"; then
    mv -- "$temporary" "$destination"
    printf 'verified package: %s\n' "$filename"
    exit 0
  fi
  : > "$temporary"
done
printf 'failed to retrieve pinned package: %s\n' "$filename" >&2
exit 1
