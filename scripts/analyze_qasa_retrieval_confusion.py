from pathlib import Path

import pandas as pd

BASE = Path("outputs/qasa")
SIZE = 512
MODEL = "jina"
CONDITIONS = ["section_isolated", "section_constrained"]

qrels = pd.read_csv(
    f"data/qasa/retrieval_units/chunk_{SIZE}/qrels.csv"
)
qrels["relevance"] = pd.to_numeric(
    qrels["relevance"], errors="coerce"
).fillna(0)

# Gold evidence-containing retrieval units for the section-bounded plan.
gold = qrels[
    (qrels["segmentation_plan"] == "section_bounded")
    & (qrels["relevance"] > 0)
]
gold_ids = gold.groupby("query_id")["retrieval_unit_id"].apply(set).to_dict()

print("Queries with gold evidence:", len(gold_ids))

rows = []

for condition in CONDITIONS:
    path = (
        BASE / "rankings" / f"chunk_{SIZE}" / condition / MODEL
        / "rankings.jsonl"
    )
    ranks = pd.read_json(path, lines=True)
    ranks["rank"] = pd.to_numeric(ranks["rank"], errors="coerce")
    ranks["score"] = pd.to_numeric(ranks["score"], errors="coerce")

    for qid, g in ranks.groupby("query_id", sort=False):
        relevant = gold_ids.get(qid, set())
        if not relevant:
            continue

        g = g.sort_values("rank").copy()
        relevant_rows = g[g["retrieval_unit_id"].isin(relevant)]

        # A relevant unit missing from this ranking is a data-integrity
        # issue worth recording, not silently treating as a normal miss.
        found_gold = set(relevant_rows["retrieval_unit_id"])
        missing_gold = relevant - found_gold

        best_rank = (
            int(relevant_rows["rank"].min())
            if not relevant_rows.empty else None
        )
        best_score = (
            float(relevant_rows["score"].max())
            if not relevant_rows.empty else None
        )

        # Focus on sections which contain gold evidence for this query.
        evidence_sections = set(
            relevant_rows["parent_section_id"].dropna()
        )

        distractors = g[
            g["parent_section_id"].isin(evidence_sections)
            & ~g["retrieval_unit_id"].isin(relevant)
        ]

        best_distractor_rank = (
            int(distractors["rank"].min())
            if not distractors.empty else None
        )
        best_distractor_score = (
            float(distractors["score"].max())
            if not distractors.empty else None
        )

        # For every evidence-containing section, did an incorrect chunk
        # appear before the first gold evidence-containing chunk?
        sections_with_gold = 0
        sections_where_distractor_wins = 0

        for section in evidence_sections:
            section_rows = g[g["parent_section_id"] == section]
            if section_rows.empty:
                continue

            section_gold = section_rows[
                section_rows["retrieval_unit_id"].isin(relevant)
            ]
            if section_gold.empty:
                continue

            sections_with_gold += 1
            first_gold_rank = section_gold["rank"].min()
            wrong_above = section_rows[
                ~section_rows["retrieval_unit_id"].isin(relevant)
                & (section_rows["rank"] < first_gold_rank)
            ]
            if not wrong_above.empty:
                sections_where_distractor_wins += 1

        rows.append({
            "query_id": qid,
            "condition": condition,
            "hit_at_5": best_rank is not None and best_rank <= 5,
            "best_gold_rank": best_rank,
            "best_gold_score": best_score,
            "best_same_section_distractor_rank": best_distractor_rank,
            "best_same_section_distractor_score": best_distractor_score,
            "score_margin_gold_minus_distractor": (
                best_score - best_distractor_score
                if best_score is not None
                and best_distractor_score is not None else None
            ),
            "evidence_sections": sections_with_gold,
            "sections_distractor_before_gold": sections_where_distractor_wins,
            "gold_ids_missing_from_ranking": len(missing_gold),
            "same_section_distractors_at_5": int(
                (
                    (g["rank"] <= 5)
                    & g["parent_section_id"].isin(evidence_sections)
                    & ~g["retrieval_unit_id"].isin(relevant)
                ).sum()
            ),
        })

long = pd.DataFrame(rows)
outdir = BASE / "analysis" / "cosine_scope"
outdir.mkdir(parents=True, exist_ok=True)
long.to_csv(outdir / "retrieval_confusion_jina_512_long.csv", index=False)

# Paired per-query comparison.
wide = long.pivot(index="query_id", columns="condition")
wide.columns = [f"{metric}__{condition}" for metric, condition in wide.columns]
wide = wide.reset_index()

wide["isolated_hits_sclc_misses"] = (
    wide["hit_at_5__section_isolated"].fillna(False)
    & ~wide["hit_at_5__section_constrained"].fillna(False)
)
wide["sclc_hits_isolated_misses"] = (
    wide["hit_at_5__section_constrained"].fillna(False)
    & ~wide["hit_at_5__section_isolated"].fillna(False)
)

# Add original query text, if the expected column exists.
queries_path = Path(
    f"data/qasa/retrieval_units/chunk_{SIZE}/queries.jsonl"
)
queries = pd.read_json(queries_path, lines=True)
text_col = next(
    (c for c in ["question", "query", "text"] if c in queries.columns),
    None,
)
if "query_id" in queries.columns and text_col:
    qtext = queries[["query_id", text_col]].drop_duplicates("query_id")
    wide = wide.merge(qtext, on="query_id", how="left")

wide.to_csv(outdir / "retrieval_confusion_jina_512_paired.csv", index=False)

# Aggregate summaries. The last metric is computed at evidence-section
# level rather than treating all chunks/pairs as independent observations.
summary = (
    long.groupby("condition")
    .agg(
        queries=("query_id", "nunique"),
        hit_at_5_rate=("hit_at_5", "mean"),
        median_best_gold_rank=("best_gold_rank", "median"),
        mean_gold_score=("best_gold_score", "mean"),
        mean_same_section_distractor_score=(
            "best_same_section_distractor_score", "mean"
        ),
        mean_gold_minus_distractor_margin=(
            "score_margin_gold_minus_distractor", "mean"
        ),
        queries_with_missing_gold=("gold_ids_missing_from_ranking",
                                   lambda s: (s > 0).sum()),
        evidence_sections=("evidence_sections", "sum"),
        sections_distractor_before_gold=(
            "sections_distractor_before_gold", "sum"
        ),
        mean_same_section_distractors_at_5=(
            "same_section_distractors_at_5", "mean"
        ),
    )
    .reset_index()
)
summary["section_distractor_first_rate"] = (
    summary["sections_distractor_before_gold"]
    / summary["evidence_sections"].replace(0, pd.NA)
)

print("\n=== SUMMARY: JINA, 512 TOKENS ===")
print(summary.round(4).to_string(index=False))

print("\n=== PAIRED HIT@5 COMPARISON ===")
print(
    "Section-isolated hits, SCLC misses:",
    int(wide["isolated_hits_sclc_misses"].sum()),
)
print(
    "SCLC hits, section-isolated misses:",
    int(wide["sclc_hits_isolated_misses"].sum()),
)

columns = [
    "query_id",
    "hit_at_5__section_isolated",
    "best_gold_rank__section_isolated",
    "hit_at_5__section_constrained",
    "best_gold_rank__section_constrained",
    "score_margin_gold_minus_distractor__section_isolated",
    "score_margin_gold_minus_distractor__section_constrained",
]
if text_col:
    columns.append(text_col)

examples = wide[wide["isolated_hits_sclc_misses"]]
print("\n=== EXAMPLES: ISOLATED HITS, SCLC MISSES (FIRST 20) ===")
if examples.empty:
    print("No such queries under this hit@5 definition.")
else:
    print(examples[[c for c in columns if c in examples.columns]]
          .head(20).to_string(index=False))

print("\nSaved to:", outdir)
