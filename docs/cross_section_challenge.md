# QASPER cross-section challenge

The cross-section challenge is an exploratory analysis separate from the frozen
primary experiment. It targets questions whose complete acceptable evidence
necessarily spans more than one top-level section.

## Collection audit and candidate discovery

Run the collection audit after `sclc prepare`:

```bash
sclc qasper-audit --config configs/base.yaml
```

Two definitions are retained:

- **Strict cross-section required:** every acceptable evidence set spans more than one top-level section.
- **Minimal cross-section required:** every smallest acceptable evidence set spans more than one top-level section, although a larger alternative may not.

The dissertation uses the strict definition and the original QASPER **test split**.
The full audit identified 302 strict candidates across 253 papers; 53 were in
the test split and were manually reviewed.

Freeze the strict test-split candidate set with:

```bash
sclc qasper-challenge --config configs/base.yaml --overwrite
```

The completed review decisions are committed at:

```text
data/subsets_cross_section_challenge/review_decisions.csv
```

The final review accepted **23 questions across 20 papers**:

- 18 questions from 15 cross-model papers;
- 5 questions from 5 Granite-extension papers.

The final dissertation reports the 18 accepted cross-model questions with Jina
at 128, 256, and 512 tokens. The Granite-extension questions remain in the
frozen review record but are not part of the reported comparable challenge results.

## Reported challenge analysis

The reported analysis uses:

- 18 accepted cross-model questions from 15 papers;
- Jina Embeddings v3;
- retrieval-unit sizes of 128, 256, and 512 tokens;
- section-isolated, section-constrained, and global contextual conditions;
- the same six principal metrics and paired document-level bootstrap procedure
  used in the primary experiment.

The two signed scope comparisons are:

- section-isolated minus section-constrained;
- section-constrained minus global.

All signed differences follow first condition minus second condition. Holm
correction is applied across these two comparisons within each retrieval-unit
size and metric.

Challenge evaluation uses `configs/cross_section_challenge.yaml` and the frozen
challenge document manifest in `data/subsets_cross_section_challenge/`.

After the three challenge conditions have been evaluated at each size, run:

```bash
sclc challenge-analyse \
  --model jina \
  --accepted-queries data/subsets_cross_section_challenge/review_decisions.csv \
  --retrieval-unit-sizes 128,256,512 \
  --config configs/cross_section_challenge.yaml \
  --overwrite
```

Outputs are written beneath:

```text
outputs/cross_section_challenge/analysis/cross_section_challenge_results/jina/
```

The challenge is exploratory and must not be combined with the primary held-out averages.
