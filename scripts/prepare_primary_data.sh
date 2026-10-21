#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${1:-configs/base.yaml}"
[[ "${CONFIG_PATH}" == /* ]] || CONFIG_PATH="${REPO_ROOT}/${CONFIG_PATH}"
command -v sclc >/dev/null 2>&1 || { echo "Install the project first: python -m pip install -e '.[dev]'" >&2; exit 2; }
[[ -f "${CONFIG_PATH}" ]] || { echo "Configuration file not found: ${CONFIG_PATH}" >&2; exit 2; }
cd "${REPO_ROOT}"
sclc prepare --config "${CONFIG_PATH}"
echo "Prepared documents are ready. The committed frozen sample manifest was not modified."
echo "To verify sampling independently: sclc profile --config ${CONFIG_PATH} && sclc sample --config ${CONFIG_PATH}"
