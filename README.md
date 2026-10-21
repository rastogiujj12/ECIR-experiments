# Section-Constrained Late Chunking (SCLC)

Research code for a controlled study of **contextual encoding scope** in dense retrieval. The central question is whether target retrieval units should be encoded independently, with context from their parent top-level section, or with context from the complete prepared document.

This is a **within-document ranking experiment**: each query is ranked only against retrieval units from its associated source document. It does not evaluate document discovery across a corpus or downstream answer generation. The results should not be read as evidence that SCLC is the best general-purpose retriever.

## Experimental conditions

All dense methods use the same embedding model within a comparison. SCLC, GLC, LMLC, and section-isolated encoding use identical section-bounded target spans; their encoding context differs.

| Condition | Target-unit segmentation | Context available when creating the representation |
|---|---|---|
| `bm25` | Continuous fixed-size units | Lexical baseline; no embeddings |
| `fixed_dense` | Continuous fixed-size units | Each unit independently encoded |
| `section_isolated` | Section-bounded units | Each target unit independently encoded |
| `section_constrained` (SCLC) | Section-bounded units | The full parent top-level section |
| `length_matched` (LMLC) | Section-bounded units | A linear window with the parent section's **canonical Granite-token count**; may cross section boundaries |
| `global` (GLC) | Section-bounded units | The complete prepared document in one contextual sequence |

The canonical Granite tokenizer defines non-overlapping target units of up to **128, 256, or 512 tokens**; short final units are retained. During encoding, model-token character offsets identify non-special content tokens overlapping the target span; their hidden states are mean-pooled and L2-normalized. Jina uses the `retrieval_query` and `retrieval_passage` adapters.

**LMLC qualification:** its window length matches the parent section in canonical Granite tokens, not necessarily the exact token count seen by every encoder. Jina's input-token counts can differ, particularly on academic documents. LMLC is therefore a length-matched *canonical-token* control, not an encoder-specific input-budget control. Its results do not by themselves isolate boundary placement from differences in context content or encoder tokenization.

Truncation/windowing is disabled for contextual inputs. If an input exceeds its configured limit, encoding fails rather than silently truncating. The configured limits are 32,768 tokens for Granite and 8,192 for Jina; the common-model evaluation sample excludes documents that cannot be encoded under both limits.

## Datasets and evaluation populations

| Dataset | Domain | Partition used | Queries | Documents/papers |
|---|---|---:|---:|---:|
| QASPER | Academic question answering | `test` | 1,134 | 353 |
| QASA | Academic question answering | `test` | 570 | 39 represented papers |
| MultiDoc2Dial | Document-grounded dialogue | `validation` | 4,388 | 272 |
| Doc2Dial | Document-grounded dialogue | `validation` | 3,919 | 272 |

The dialogue datasets use section spans from their grounded source documents, not dialogue turns. MultiDoc2Dial's released v1.0 test dialogue file is a dummy, so the frozen validation partition is used; Doc2Dial also uses validation. These validation results are **not** held-out test-set results and should be described as such. Query/candidate evaluation is restricted to the source document associated with each query.

The two academic and two dialogue datasets were retained to examine whether contextual-scope behaviour varies both within and across domains. The populations above are those used in the reported experiment; preserve the selected-document manifests and source-document caches when attempting exact replication.

## Main result

The primary metric is **nDCG@5**. The updated paired document-bootstrap analysis compares SCLC against section-isolated encoding, LMLC, and GLC using 10,000 resamples, 95% percentile confidence intervals, two-sided recentered-null bootstrap p-values with a plus-one correction, and Holm adjustment within each dataset across 18 planned contrasts (2 models × 3 unit sizes × 3 comparators; adjusted `p < .05`).

- SCLC has higher mean nDCG@5 than GLC in all 24 dataset–model–size settings; 19/24 contrasts are significant after Holm correction.
- SCLC has higher mean nDCG@5 than LMLC in all 24 settings; 20/24 contrasts are significant after correction.
- Section-isolated encoding has higher mean nDCG@5 than SCLC in 22/24 settings; 10/24 differences significantly favour section-isolated encoding, and none significantly favour SCLC.

The 24 settings are related experimental conditions, **not 24 independent replications**. Counts across settings are descriptive. BM25 and fixed dense are useful baselines, but they use continuous target units, so comparisons with section-bounded methods also reflect segmentation differences. The experiment shows retrieval-effectiveness differences; it does not establish a causal mechanism such as semantic hijacking or semantic dilution.

Compact result files are in [`results/`](results/): all 72 paired contrasts with effect sizes and adjusted p-values, pairing diagnostics, and the analysis protocol.

## Installation

Python 3.11 or 3.12 is supported.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Prepare the datasets

Use the configuration for the dataset being prepared; configurations keep processed data and outputs in separate directories.

### QASPER

The frozen QASPER sample and coding files are committed with the repository. Prepare the data without replacing the frozen sample manifest:

```bash
./scripts/prepare_primary_data.sh configs/base.yaml
```

### QASA

```bash
python scripts/download_qasa.py
python scripts/prepare_qasa.py configs/qasa.yaml
sclc profile --config configs/qasa.yaml
sclc sample --config configs/qasa.yaml
```

QASA paper availability may depend on cached/public full-text sources. For exact replication, preserve the selected-document manifest and the paper cache used for the reported run.

### MultiDoc2Dial

```bash
python scripts/prepare_multidoc2dial.py configs/multidoc2dial.yaml --split validation
```

### Doc2Dial

```bash
python scripts/prepare_doc2dial.py configs/doc2dial.yaml --split validation
```

The two dialogue preparation scripts use publisher-provided source-document structure and set the fixed validation population. Keep each dataset's generated files under its config-specific directory.

## Run the primary experiment

The orchestration script runs BM25 and all five dense conditions, including LMLC, at all three unit sizes, then runs the paired comparisons. Run one model at a time:

```bash
./scripts/run_primary_experiment.sh configs/base.yaml granite
./scripts/run_primary_experiment.sh configs/base.yaml jina
```

For other datasets, substitute `configs/qasa.yaml`, `configs/multidoc2dial.yaml`, or `configs/doc2dial.yaml` after preparing that dataset. Example:

```bash
./scripts/run_primary_experiment.sh configs/doc2dial.yaml granite
./scripts/run_primary_experiment.sh configs/doc2dial.yaml jina
```

The script uses each config's paths and evaluation split. Runs require the model dependencies and sufficient memory for the contextual sequences; over-limit inputs are rejected, not truncated. Generated model caches, encodings, rankings, evaluation outputs, and bootstrap arrays are not committed.

## Other analyses and checks

Retrieval-unit-size analysis:

```bash
sclc retrieval-unit-size \
  --model granite \
  --retrieval-unit-sizes 128,256,512 \
  --config configs/base.yaml \
  --overwrite
```

Signed differences follow **first condition minus second condition**. The exploratory cross-section challenge and query-type coding procedures are documented in [`docs/cross_section_challenge.md`](docs/cross_section_challenge.md) and [`docs/query_type_coding_guide.md`](docs/query_type_coding_guide.md).

Before creating a release:

```bash
pytest
ruff check .
mypy src
python scripts/check_release.py
```

Use `python scripts/capture_environment.py` on the experiment machine to record package versions, hardware/CUDA details, model metadata, cache references and frozen-input hashes. The original experiment configuration uses model repository IDs rather than immutable checkpoint commit IDs; checkpoint-level replication may therefore depend on model revisions available in the local cache.

## Repository layout

```text
configs/          Dataset-specific experiment configurations
data/             Frozen sample and coding artefacts (not the generated corpus)
docs/             Method, coding, and analysis documentation
results/          Compact paired statistical outputs reported in the paper
reproducibility/  Frozen-input and environment records
scripts/          Dataset preparation, experiment orchestration, and release helpers
src/sclc/         Experiment implementation
tests/            Regression tests
```

## Citation and acknowledgements

If you use this code, cite the associated paper when its publication details are available. Please check dataset and model licences/terms at their original sources before redistributing raw data or model files.

## Implementation notes

- MultiDoc2Dial overlapping structural spans are normalized into disjoint document-ordered intervals, covered by a regression test.
- Dataset-specific configurations isolate processed data, subsets, retrieval units, encodings, rankings, evaluations and analyses.
- Doc2Dial uses publisher-provided document text, section spans and dialogue-grounding references. Its validation split is the fixed evaluation population.
- For QASA records without arXiv identifiers, the preparer can discover public PDFs via Semantic Scholar. Valid locally cached S2ORC PDFs take priority, avoiding unnecessary network lookups and API-rate-limit failures.
