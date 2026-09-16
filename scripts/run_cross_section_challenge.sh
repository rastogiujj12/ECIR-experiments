#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${1:-configs/cross_section_challenge.yaml}"
MODEL="${2:-jina}"
ACCEPTED_QUERIES="${3:-data/subsets_cross_section_challenge/review_decisions.csv}"
[[ "${CONFIG_PATH}" == /* ]] || CONFIG_PATH="${REPO_ROOT}/${CONFIG_PATH}"
[[ "${ACCEPTED_QUERIES}" == /* ]] || ACCEPTED_QUERIES="${REPO_ROOT}/${ACCEPTED_QUERIES}"
[[ "${MODEL}" == "granite" || "${MODEL}" == "jina" ]] || { echo "Usage: $0 [config-path] {granite|jina} [review-decisions.csv]" >&2; exit 2; }
command -v sclc >/dev/null 2>&1 || { echo "Install the project first: python -m pip install -e '.[dev]'" >&2; exit 2; }
[[ -f "${CONFIG_PATH}" && -f "${ACCEPTED_QUERIES}" ]] || { echo "Required config/review file missing." >&2; exit 2; }
cd "${REPO_ROOT}"
[[ -f data/processed/documents.jsonl ]] || { echo "Prepared QASPER documents are missing." >&2; exit 2; }
[[ -f data/subsets_cross_section_challenge/selected_documents.csv ]] || { echo "Frozen challenge document manifest is missing." >&2; exit 2; }
SIZES=(128 256 512)
CONDITIONS=(section_isolated section_constrained global)
for SIZE in "${SIZES[@]}"; do
  sclc build-units --retrieval-unit-size "${SIZE}" --config "${CONFIG_PATH}"
  for CONDITION in "${CONDITIONS[@]}"; do
    sclc encode --condition "${CONDITION}" --model "${MODEL}" --retrieval-unit-size "${SIZE}" --config "${CONFIG_PATH}"
    sclc retrieve --condition "${CONDITION}" --model "${MODEL}" --retrieval-unit-size "${SIZE}" --config "${CONFIG_PATH}"
    sclc evaluate --condition "${CONDITION}" --model "${MODEL}" --retrieval-unit-size "${SIZE}" --config "${CONFIG_PATH}"
  done
done
sclc challenge-analyse --model "${MODEL}" --accepted-queries "${ACCEPTED_QUERIES}" --retrieval-unit-sizes 128,256,512 --config "${CONFIG_PATH}" --overwrite
