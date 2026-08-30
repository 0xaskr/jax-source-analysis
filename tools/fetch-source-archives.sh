#!/usr/bin/env bash

set -u

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="${project_root}/source-archives.lock"
failures=()
scratch="$(mktemp -d)"

cleanup() {
  rm -rf -- "${scratch}"
}
trap cleanup EXIT

cd "${project_root}"

while IFS='|' read -r category path url expected_sha strip_components pinned_by; do
  case "${category}" in
    ''|'#'*) continue ;;
  esac

  echo "[${category}] ${path}"
  if [[ -f "${path}/.source-origin" ]]; then
    echo "  already extracted"
    continue
  fi
  if [[ -e "${path}" ]]; then
    failures+=("${path}: exists without .source-origin")
    continue
  fi

  archive="${scratch}/$(basename "${url}")"
  if ! curl --fail --location --retry 3 --output "${archive}" "${url}"; then
    failures+=("${path}: download failed")
    continue
  fi

  actual_sha="$(sha256sum "${archive}" | awk '{print $1}')"
  if [[ "${actual_sha}" != "${expected_sha}" ]]; then
    failures+=("${path}: sha256 mismatch")
    continue
  fi

  mkdir -p -- "${path}"
  if ! tar -xf "${archive}" -C "${path}" --strip-components="${strip_components}"; then
    failures+=("${path}: extraction failed")
    continue
  fi

  printf 'url=%s\nsha256=%s\npinned_by=%s\n' \
    "${url}" "${expected_sha}" "${pinned_by}" > "${path}/.source-origin"
done < "${manifest}"

if ((${#failures[@]})); then
  echo
  echo "Failed archives:"
  printf '  - %s\n' "${failures[@]}"
  exit 1
fi

echo
echo "All source archives are extracted."
