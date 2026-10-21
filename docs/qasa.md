# QASA integration

This branch adds a QASA preparation path without changing the existing QASPER preparation path.

## Source

The official QASA repository releases the answerable v1.1 JSON at:

`https://github.com/lgresearch/QASA/raw/refs/heads/main/data/testset_answerable_1554_v1.1.json`

The repository notes that v1.1 adds the URL needed to retrieve each paper's full text. The official helper retrieves paper HTML from ar5iv. The adapter keeps those paper downloads in `data/qasa/papers/`. When the current ar5iv rendering no longer reproduces the stored evidence, the adapter checks historical arXiv versions and can reconstruct a paper from its historical source archive, preserving source paragraph boundaries before falling back to PDF reconstruction.

## Why the adapter is separate

The existing `reconstruct_document()` function is QASPER-specific. QASA has a different QA schema, so this integration converts QASA into the same internal `DocumentRecord` schema before the existing chunking, LMLC, encoding, retrieval, evaluation, and statistical layers are used.

## Preparation workflow

1. Download the official JSON:

```bash
python scripts/download_qasa.py
```

2. Inspect the actual released JSON schema before preparing the full corpus:

```bash
python scripts/probe_qasa.py data/qasa/raw/testset_answerable_1554_v1.1.json
```

3. Smoke-test the adapter on a small number of papers:

```bash
python scripts/prepare_qasa.py configs/qasa.yaml --limit-papers 2
```

4. Inspect `data/qasa/processed/evidence_alignment.csv`. The key check is that every evidence item for a question is aligned to the reconstructed document. Exact matches are preferred, with conservative lexical fuzzy recovery used only for formatting/version drift. Duplicate exact matches are treated as ambiguous rather than silently assigning relevance to every occurrence. Questions with incomplete or ambiguous evidence alignment are excluded from retrieval evaluation.

5. Prepare the complete answerable corpus:

```bash
python scripts/prepare_qasa.py configs/qasa.yaml
```

6. Profile both embedding tokenizers and select the common cross-model corpus:

```bash
sclc profile --config configs/qasa.yaml
sclc sample --config configs/qasa.yaml
```

7. Construct 128/256/512-token retrieval units and proceed through the existing encode/retrieve/evaluate/compare commands.

## Current feasibility constraint

The adapter has been validated against a two-paper smoke test using the released QASA v1.1 schema and reconstructed paper records. The development runtime itself does not provide reliable outbound access to the full 4.43 MB release, so the full 1,554-paper corpus must still be prepared on an internet-enabled machine. Start with the schema probe and two-paper smoke test before launching the complete preparation run.

The adapter deliberately avoids semantic evidence matching. Exact normalized paragraph matching is preferred after deterministic source reconstruction, with a conservative lexical fallback based on exact token anchors and local sequence similarity for formatting/version drift. Retrieved relevance is never inferred from embeddings or an LLM. Historical source reconstruction uses the original TeX paragraph boundaries and local bibliography text where available. If the official file exposes evidence as context IDs rather than text, the adapter supports resolving IDs through a record-local `ctxs`/`contexts`/`passages` collection.


## Parser and alignment details

The QASA repository's reference helper selects paragraph nodes by their ar5iv
paragraph ids (for example, ids beginning with `S1.p` or `S1.SS...`) rather
than by paragraph ancestry alone. The adapter follows this convention and
groups nested-subsection paragraphs under their top-level `S<number>` section.
Gold evidence alignment is restricted to body paragraph nodes; title and abstract text remain in the retrieval document but are not treated as QASA evidence targets. Version scoring is paragraph-local rather than document-concatenated, preventing accidental cross-paragraph string matches from selecting the wrong paper version.

Cached pages are validated before reuse; pages without top-level sections and
QASA paragraph ids are discarded and re-fetched from the primary ar5iv URL,
with `ar5iv.org` as an alternate fallback.


## Historical source reconstruction

When current or versioned ar5iv does not reproduce the stored QASA evidence, the adapter checks the paper's arXiv version history and downloads the matching `e-print` source archive. The source tree is expanded, `\input`/`\include` dependencies are resolved, and standard TeX is converted to the minimal QASA section/paragraph representation. Pandoc is used when available; a deterministic TeX fallback handles common formatting commands, citations, and math source. Source archives are cached under `data/qasa/papers/source/`.


### Alternate representation selection

When multiple representations of the same paper are available, QASA preparation ranks them by conservative lexical evidence recoverability before exact string coverage. This avoids selecting a PDF reconstruction solely because it preserves a few exact strings when current ar5iv contains substantially more of the benchmark evidence with harmless citation/footnote/math formatting drift.

## Non-arXiv QASA papers

QASA v1.1 includes a small non-arXiv subset with `s2orc_id` but no `arxiv_id`. The adapter preserves the S2ORC identifier and, when needed, queries Semantic Scholar's paper-details API by `CorpusId:<s2orc_id>` to discover an open-access PDF. That PDF is validated and passed through the same deterministic PDF-to-paragraph reconstruction and evidence alignment used by the arXiv fallback. The discovered PDF is cached under `data/qasa/papers/<paper_id>.s2orc.pdf`.
