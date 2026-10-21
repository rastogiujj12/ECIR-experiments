#!/usr/bin/env bash
set -euo pipefail

# Full QASA retrieval experiment.
# Run from the repository root, e.g.:
#   bash scripts/run_qasa_full.sh
#
# Safe to rerun: existing outputs are reused unless the underlying project
# itself requires overwrite/rebuild. No --overwrite flags are used here.

CONFIG="configs/qasa.yaml"
SIZES=(128 256 512)
DENSE_CONDITIONS=(fixed_dense section_isolated section_constrained global length_matched)
MODELS=(granite jina)

log() {
  echo
  echo "================================================================"
  echo "$*"
  echo "================================================================"
}

[[ -f "$CONFIG" ]] || { echo "Missing $CONFIG" >&2; exit 2; }
command -v sclc >/dev/null 2>&1 || {
  echo "sclc is not installed/available. Activate your venv and install the project first." >&2
  exit 2
}

[[ -f data/qasa/processed/qasa_eligibility_manifest.json ]] || {
  echo "Frozen QASA eligibility manifest is missing." >&2
  exit 2
}
[[ -f data/qasa/processed/documents.jsonl ]] || {
  echo "Prepared QASA documents are missing. Run prepare_qasa.py first." >&2
  exit 2
}

log "1/4: PROFILE QASA DOCUMENTS"
sclc profile --config "$CONFIG"

log "2/4: SELECT ALL ELIGIBLE QASA PAPERS"
sclc sample --config "$CONFIG"

for SIZE in "${SIZES[@]}"; do
  log "3/4: BUILD ${SIZE}-TOKEN RETRIEVAL UNITS"
  sclc build-units --retrieval-unit-size "$SIZE" --config "$CONFIG"

  log "${SIZE}-TOKEN: BM25"
  sclc encode --condition bm25 --retrieval-unit-size "$SIZE" --config "$CONFIG"
  sclc retrieve --condition bm25 --retrieval-unit-size "$SIZE" --config "$CONFIG"
  sclc evaluate --condition bm25 --retrieval-unit-size "$SIZE" --config "$CONFIG"

  for MODEL in "${MODELS[@]}"; do
    for CONDITION in "${DENSE_CONDITIONS[@]}"; do
      log "${SIZE}-TOKEN: ${MODEL} / ${CONDITION}"
      sclc encode \
        --condition "$CONDITION" \
        --model "$MODEL" \
        --retrieval-unit-size "$SIZE" \
        --config "$CONFIG"

      sclc retrieve \
        --condition "$CONDITION" \
        --model "$MODEL" \
        --retrieval-unit-size "$SIZE" \
        --config "$CONFIG"

      sclc evaluate \
        --condition "$CONDITION" \
        --model "$MODEL" \
        --retrieval-unit-size "$SIZE" \
        --config "$CONFIG"
    done

    log "${SIZE}-TOKEN: BUILT-IN PAIRWISE COMPARISONS / ${MODEL}"
    sclc compare \
      --model "$MODEL" \
      --retrieval-unit-size "$SIZE" \
      --config "$CONFIG"
  done
done

log "4/4: RETRIEVAL-UNIT-SIZE ANALYSIS (BUILT-IN FOUR DENSE CONDITIONS)"
for MODEL in "${MODELS[@]}"; do
  sclc retrieval-unit-size \
    --model "$MODEL" \
    --retrieval-unit-sizes 128,256,512 \
    --config "$CONFIG"
 done

log "QASA RUN COMPLETE"
echo "Prepared papers: $(wc -l < data/qasa/subsets/selected_documents.csv) rows including header"
echo "Frozen eligible questions: $(wc -l < data/qasa/processed/qasa_eligible_questions.jsonl)"
echo ""
echo "Dense evaluations written under: outputs/qasa/evaluation/"
echo "Rankings written under:          outputs/qasa/rankings/"
echo "Encodings written under:         outputs/qasa/encodings/"
echo "Analysis written under:          data/qasa/processed/../analysis or outputs/qasa/analysis as configured"
echo ""
echo "NOTE: length_matched is fully encoded/retrieved/evaluated above."
echo "      The built-in compare/retrieval-unit-size analysis currently excludes length_matched."
