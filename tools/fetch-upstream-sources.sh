#!/usr/bin/env bash

set -u

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="${project_root}/upstream-sources.lock"
failures=()
selected_paths=("$@")

is_selected() {
  local candidate="$1"
  local selected
  if ((${#selected_paths[@]} == 0)); then
    return 0
  fi
  for selected in "${selected_paths[@]}"; do
    if [[ "${candidate}" == "${selected}" ]]; then
      return 0
    fi
  done
  return 1
}

cd "${project_root}"

while IFS='|' read -r category path url ref pinned_by sparse_paths checkout_policy; do
  case "${category}" in
    ''|'#'*) continue ;;
  esac
  if ! is_selected "${path}"; then
    continue
  fi
  if [[ "${checkout_policy:-eager}" == "lazy" \
      && ${#selected_paths[@]} -eq 0 ]]; then
    echo "[lazy] ${path} (registered, not initialized)"
    continue
  fi

  echo "[${category}] ${path} @ ${ref}"

  if ! git config -f .gitmodules --get-regexp '^submodule\..*\.path$' 2>/dev/null \
      | awk '{print $2}' | grep -Fxq "${path}"; then
    if [[ ! -e "${path}/.git" && ! -f "${path}/.git" ]]; then
      if ! git clone --filter=blob:none --no-checkout --depth 1 \
          "${url}" "${path}"; then
        failures+=("${path}: clone failed")
        continue
      fi
    fi

    if ! resolved_new="$(git -C "${path}" rev-parse "${ref}^{commit}" 2>/dev/null)"; then
      if ! git -C "${path}" fetch --depth 1 origin "${ref}"; then
        failures+=("${path}: could not fetch ${ref}")
        continue
      fi
      resolved_new="$(git -C "${path}" rev-parse "FETCH_HEAD^{commit}")"
    fi

    if [[ -n "${sparse_paths}" ]]; then
      read -r -a sparse_path_array <<< "${sparse_paths}"
      if ! git -C "${path}" sparse-checkout set "${sparse_path_array[@]}"; then
        failures+=("${path}: sparse checkout configuration failed")
        continue
      fi
    fi

    if ! git -C "${path}" -c advice.detachedHead=false \
          checkout --detach "${resolved_new}" \
        || ! git config -f .gitmodules "submodule.${path}.path" "${path}" \
        || ! git config -f .gitmodules "submodule.${path}.url" "${url}" \
        || ! git update-index --add --cacheinfo \
          "160000,${resolved_new},${path}" \
        || ! git submodule absorbgitdirs -- "${path}" \
        || ! git submodule init -- "${path}"; then
      failures+=("${path}: submodule registration failed")
      continue
    fi
  elif [[ ! -e "${path}/.git" && ! -f "${path}/.git" ]]; then
    if [[ -n "${sparse_paths}" ]]; then
      clone_args=(--filter=blob:none --sparse --no-checkout --depth 1)
      if [[ ! "${ref}" =~ ^[0-9a-f]{40}$ ]]; then
        clone_args+=(--branch "${ref}")
      fi
      if ! git clone "${clone_args[@]}" "${url}" "${path}"; then
        failures+=("${path}: sparse clone failed")
        continue
      fi
      if ! resolved_sparse="$(git -C "${path}" rev-parse "${ref}^{commit}" 2>/dev/null)"; then
        if ! git -C "${path}" fetch --depth 1 origin "${ref}"; then
          failures+=("${path}: sparse clone could not fetch ${ref}")
          continue
        fi
        resolved_sparse="$(git -C "${path}" rev-parse "FETCH_HEAD^{commit}")"
      fi
      read -r -a sparse_path_array <<< "${sparse_paths}"
      if ! git -C "${path}" sparse-checkout set "${sparse_path_array[@]}" \
          || ! git -C "${path}" -c advice.detachedHead=false \
            checkout --detach "${resolved_sparse}" \
          || ! git submodule absorbgitdirs -- "${path}" \
          || ! git submodule init -- "${path}"; then
        failures+=("${path}: sparse checkout initialization failed")
        continue
      fi
    elif ! git submodule update --init --depth 1 -- "${path}"; then
      failures+=("${path}: initialization failed")
      continue
    fi
  fi

  if [[ -n "${sparse_paths}" ]]; then
    read -r -a sparse_path_array <<< "${sparse_paths}"
    if ! git -C "${path}" sparse-checkout set "${sparse_path_array[@]}"; then
      failures+=("${path}: sparse checkout configuration failed")
      continue
    fi
  fi

  current="$(git -C "${path}" rev-parse HEAD)"
  if resolved_local="$(git -C "${path}" rev-parse "${ref}^{commit}" 2>/dev/null)" \
      && [[ "${current}" == "${resolved_local}" ]]; then
    git add -- ".gitmodules" "${path}"
    continue
  fi

  if ! git -C "${path}" fetch --depth 1 origin "${ref}"; then
    failures+=("${path}: could not fetch ${ref}")
    continue
  fi

  resolved="$(git -C "${path}" rev-parse "FETCH_HEAD^{commit}")"
  if ! git -C "${path}" -c advice.detachedHead=false checkout --detach "${resolved}"; then
    failures+=("${path}: checkout failed")
    continue
  fi

  git add -- ".gitmodules" "${path}"
done < "${manifest}"

if ((${#failures[@]})); then
  echo
  echo "Failed sources:"
  printf '  - %s\n' "${failures[@]}"
  exit 1
fi

echo
echo "All manifest sources are initialized and pinned."
