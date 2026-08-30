#!/usr/bin/env bash

set -u

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="${project_root}/upstream-sources.lock"
archive_manifest="${project_root}/source-archives.lock"
failures=()
initialized_count=0
lazy_count=0

declare -A manifest_paths=()
declare -A module_paths=()

cd "${project_root}"

while IFS='|' read -r category path url ref pinned_by sparse_paths checkout_policy; do
  case "${category}" in
    ''|'#'*) continue ;;
  esac

  if [[ -n "${manifest_paths[${path}]:-}" ]]; then
    failures+=("${path}: duplicate manifest entry")
    continue
  fi
  manifest_paths["${path}"]="${url}"

  gitlink="$(git ls-files --stage -- "${path}" | awk '$1 == "160000" {print $2}')"
  if [[ -z "${gitlink}" ]]; then
    failures+=("${path}: no gitlink in the analysis repository index")
    continue
  fi

  if [[ ! -e "${path}/.git" && ! -f "${path}/.git" ]]; then
    if [[ "${checkout_policy:-eager}" == "lazy" ]]; then
      ((lazy_count += 1))
      continue
    fi
    failures+=("${path}: submodule is not initialized")
    continue
  fi
  ((initialized_count += 1))

  current="$(git -C "${path}" rev-parse HEAD 2>/dev/null || true)"
  if [[ "${current}" != "${gitlink}" ]]; then
    failures+=("${path}: HEAD ${current:-missing} differs from gitlink ${gitlink}")
  fi

  if resolved="$(git -C "${path}" rev-parse --verify \
      "${ref}^{commit}" 2>/dev/null)"; then
    if [[ "${current}" != "${resolved}" ]]; then
      failures+=("${path}: HEAD does not match manifest ref ${ref}")
    fi
  fi

  if [[ -n "${sparse_paths}" ]]; then
    for sparse_path in ${sparse_paths}; do
      if [[ ! -e "${path}/${sparse_path}" ]]; then
        failures+=("${path}: sparse source path ${sparse_path} is missing")
      fi
    done
  fi
done < "${manifest}"

while read -r key path; do
  section="${key%.path}"
  module_paths["${path}"]=1
  if [[ -z "${manifest_paths[${path}]:-}" ]]; then
    failures+=("${path}: .gitmodules entry is absent from upstream-sources.lock")
    continue
  fi
  configured_url="$(git config -f .gitmodules --get "${section}.url")"
  if [[ "${configured_url}" != "${manifest_paths[${path}]}" ]]; then
    failures+=("${path}: .gitmodules URL differs from manifest")
  fi
done < <(git config -f .gitmodules --get-regexp '^submodule\..*\.path$')

for path in "${!manifest_paths[@]}"; do
  if [[ -z "${module_paths[${path}]:-}" ]]; then
    failures+=("${path}: manifest entry is absent from .gitmodules")
  fi
done

archive_count=0
while IFS='|' read -r category path url expected_sha strip_components pinned_by; do
  case "${category}" in
    ''|'#'*) continue ;;
  esac
  ((archive_count += 1))
  marker="${path}/.source-origin"
  if [[ ! -f "${marker}" ]]; then
    failures+=("${path}: exact source archive is not extracted")
    continue
  fi
  marker_sha="$(awk -F= '$1 == "sha256" {print $2}' "${marker}")"
  if [[ "${marker_sha}" != "${expected_sha}" ]]; then
    failures+=("${path}: source archive marker has the wrong sha256")
  fi
done < "${archive_manifest}"

if ((${#failures[@]})); then
  echo "Source verification failed:"
  printf '  - %s\n' "${failures[@]}"
  exit 1
fi

echo "Verified ${#manifest_paths[@]} registered submodules (${initialized_count} initialized, ${lazy_count} lazy) and ${archive_count} exact source archive."
