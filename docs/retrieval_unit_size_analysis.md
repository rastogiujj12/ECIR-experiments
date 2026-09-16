# Retrieval-unit-size analysis

The dissertation evaluates all primary retrieval conditions at canonical target
sizes of **128, 256, and 512 tokens**. Retrieval-unit size is an experimental
factor; all three sizes are evaluated directly rather than selected by a preliminary size-selection step.

For each embedding model, the analysis uses the frozen paper sample and reports:

- mean performance at each retrieval-unit size;
- the three dense comparisons used in the dissertation;
- per-query contextual-scope effects;
- interactions testing whether a scope effect changes between retrieval-unit sizes.

Granite results are available for the cross-model core, the Granite extension,
and their union. Jina is restricted to the cross-model core because complete
papers above Jina's configured context limit are not truncated or windowed.

## Run the experiment for one model

The convenience script runs the complete primary workflow for one model,
including BM25, all four dense conditions, pairwise comparisons, and the
retrieval-unit-size analysis:

```bash
./scripts/run_primary_experiment.sh configs/base.yaml granite
./scripts/run_primary_experiment.sh configs/base.yaml jina
```

## Run only the size analysis

After all required evaluations exist:

```bash
sclc retrieval-unit-size \
  --model granite \
  --retrieval-unit-sizes 128,256,512 \
  --config configs/base.yaml \
  --overwrite
```

Repeat with `--model jina` for the common cross-model sample.

## Outputs

Outputs are written to:

```text
outputs/analysis/retrieval_unit_size/<model>/
```

The main files are:

- `summary_by_retrieval_unit_size.csv`: mean metrics by condition, size, analysis set,
  and sample scope;
- `comparisons_within_retrieval_unit_size.csv`: paired document-level bootstrap
  comparisons at each size;
- `query_scope_effects.csv`: per-query section-isolated minus
  section-constrained and section-constrained minus global effects;
- `scope_interactions_across_retrieval_unit_sizes.csv`: tests of whether those scope
  effects change between 128, 256, and 512 tokens;
- `manifest.json`: configuration, source fingerprints, and output metadata.

All signed differences follow **first condition minus second condition**. A
positive difference favours the first named condition; a negative difference
favours the second.

Fixed-rank metrics are retained because they describe early-ranking behaviour,
but larger retrieval units return more text at the same rank cutoff. The
fixed-token-budget metrics therefore provide the more controlled comparison
across retrieval-unit sizes.
