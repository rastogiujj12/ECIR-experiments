#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${1:-configs/base.yaml}"
MODEL="${2:-}"
[[ "${CONFIG_PATH}" == /* ]] || CONFIG_PATH="${REPO_ROOT}/${CONFIG_PATH}"
[[ "${MODEL}" == "granite" || "${MODEL}" == "jina" ]] || { echo "Usage: $0 [config-path] {granite|jina}" >&2; exit 2; }
command -v sclc >/dev/null 2>&1 || { echo "Install the project first: python -m pip install -e '.[dev]'" >&2; exit 2; }
[[ -f "${CONFIG_PATH}" ]] || { echo "Configuration file not found: ${CONFIG_PATH}" >&2; exit 2; }
cd "${REPO_ROOT}"
[[ -f data/processed/documents.jsonl ]] || { echo "Prepared QASPER documents are missing. Run ./scripts/prepare_primary_data.sh first." >&2; exit 2; }
[[ -f data/subsets/selected_documents.csv ]] || { echo "Frozen sample manifest is missing." >&2; exit 2; }
[[ -f data/retrieval_units/query_types.csv ]] || { echo "Frozen query-type labels are missing." >&2; exit 2; }
SIZES=(128 256 512)
DENSE_CONDITIONS=(fixed_dense section_isolated section_constrained global)
for SIZE in "${SIZES[@]}"; do
  echo "=== Retrieval-unit size: ${SIZE} tokens ==="
  sclc build-units --retrieval-unit-size "${SIZE}" --config "${CONFIG_PATH}"
  sclc encode --condition bm25 --retrieval-unit-size "${SIZE}" --config "${CONFIG_PATH}"
  sclc retrieve --condition bm25 --retrieval-unit-size "${SIZE}" --config "${CONFIG_PATH}"
  sclc evaluate --condition bm25 --retrieval-unit-size "${SIZE}" --config "${CONFIG_PATH}"
  for CONDITION in "${DENSE_CONDITIONS[@]}"; do
    sclc encode --condition "${CONDITION}" --model "${MODEL}" --retrieval-unit-size "${SIZE}" --config "${CONFIG_PATH}"
    sclc retrieve --condition "${CONDITION}" --model "${MODEL}" --retrieval-unit-size "${SIZE}" --config "${CONFIG_PATH}"
    sclc evaluate --condition "${CONDITION}" --model "${MODEL}" --retrieval-unit-size "${SIZE}" --config "${CONFIG_PATH}"
  done
  sclc compare --model "${MODEL}" --retrieval-unit-size "${SIZE}" --config "${CONFIG_PATH}" --overwrite
done
sclc retrieval-unit-size --model "${MODEL}" --retrieval-unit-sizes 128,256,512 --config "${CONFIG_PATH}" --overwrite
