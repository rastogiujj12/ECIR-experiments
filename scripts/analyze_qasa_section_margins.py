from pathlib import Path

import numpy as np
import pandas as pd

SIZE = 512
MODEL = "jina"
CONDITIONS = ["section_isolated", "section_constrained"]

base = Path("outputs/qasa")
outdir = base / "analysis" / "cosine_scope"
outdir.mkdir(parents=True, exist_ok=True)

qrels = pd.read_csv(
    f"data/qasa/retrieval_units/chunk_{SIZE}/qrels.csv"
)

gold = qrels[
    (qrels["segmentation_plan"] == "section_bounded")
    & (pd.to_numeric(qrels["relevance"], errors="coerce") > 0)
]
gold_ids = (
    gold.groupby("query_id")["retrieval_unit_id"]
    .apply(set)
    .to_dict()
)

records = []

for condition in CONDITIONS:
    path = (
        base / "rankings" / f"chunk_{SIZE}" / condition / MODEL
        / "rankings.jsonl"
    )
    rankings = pd.read_json(path, lines=True)
    rankings["rank"] = pd.to_numeric(rankings["rank"])
    rankings["score"] = pd.to_numeric(rankings["score"])

    for qid, query_rows in rankings.groupby("query_id", sort=False):
        relevant = gold_ids.get(qid, set())
        if not relevant:
            continue

        # Find the sections containing gold units for this query.
        gold_rows = query_rows[
            query_rows["retrieval_unit_id"].isin(relevant)
        ]

        for section in gold_rows["parent_section_id"].dropna().unique():
            section_rows = query_rows[
                query_rows["parent_section_id"] == section
            ]

            section_gold = section_rows[
                section_rows["retrieval_unit_id"].isin(relevant)
            ]
            distractors = section_rows[
                ~section_rows["retrieval_unit_id"].isin(relevant)
            ]

            if section_gold.empty or distractors.empty:
                continue

            best_gold_rank = int(section_gold["rank"].min())
            best_gold_score = float(section_gold["score"].max())
            best_distractor_rank = int(distractors["rank"].min())
            best_distractor_score = float(distractors["score"].max())

            records.append({
                "query_id": qid,
                "document_id": str(query_rows["document_id"].iloc[0]),
                "section_id": str(section),
                "condition": condition,
                "gold_rank": best_gold_rank,
                "distractor_rank": best_distractor_rank,
                "gold_hit_at_5": best_gold_rank <= 5,
                "distractor_before_gold":
                    best_distractor_rank < best_gold_rank,
                "gold_minus_distractor_score":
                    best_gold_score - best_distractor_score,
                "gold_score": best_gold_score,
                "distractor_score": best_distractor_score,
            })

df = pd.DataFrame(records)
df.to_csv(outdir / "section_margin_details.csv", index=False)

keys = ["query_id", "document_id", "section_id"]
paired = df.pivot(
    index=keys,
    columns="condition",
    values=[
        "gold_hit_at_5",
        "distractor_before_gold",
        "gold_minus_distractor_score",
        "gold_rank",
        "distractor_rank",
    ],
)

paired.columns = [
    f"{metric}__{condition}" for metric, condition in paired.columns
]
paired = paired.reset_index()

print("\n=== SAME-SECTION COMPARISON ===")
print("Matched query-section pairs:", len(paired))

for condition in CONDITIONS:
    print(f"\n{condition}")
    for metric in [
        "gold_hit_at_5",
        "distractor_before_gold",
        "gold_minus_distractor_score",
    ]:
        col = f"{metric}__{condition}"
        print(f"  {metric}: {paired[col].mean():.4f}")

paired["sclc_minus_isolated_margin"] = (
    paired["gold_minus_distractor_score__section_constrained"]
    - paired["gold_minus_distractor_score__section_isolated"]
)
paired["sclc_minus_isolated_confusion"] = (
    paired["distractor_before_gold__section_constrained"].astype(float)
    - paired["distractor_before_gold__section_isolated"].astype(float)
)
paired["sclc_minus_isolated_gold_hit"] = (
    paired["gold_hit_at_5__section_constrained"].astype(float)
    - paired["gold_hit_at_5__section_isolated"].astype(float)
)

print("\n=== SCLC MINUS SECTION-ISOLATED ===")
for metric in [
    "sclc_minus_isolated_margin",
    "sclc_minus_isolated_confusion",
    "sclc_minus_isolated_gold_hit",
]:
    print(f"  {metric}: {paired[metric].mean():.4f}")

# Document-level paired bootstrap: sections and queries from the same paper
# are kept together rather than treated as independent observations.
rng = np.random.default_rng(42)
doc_ids = paired["document_id"].unique()
metrics = [
    "sclc_minus_isolated_margin",
    "sclc_minus_isolated_confusion",
    "sclc_minus_isolated_gold_hit",
]

print("\n=== DOCUMENT-LEVEL BOOTSTRAP 95% INTERVALS ===")
for metric in metrics:
    by_doc = paired.groupby("document_id")[metric].mean()
    vals = pd.to_numeric(
        by_doc.reindex(doc_ids), errors="coerce"
    ).to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        continue

    samples = rng.choice(
        vals, size=(10000, len(vals)), replace=True
    ).mean(axis=1)

    low, high = np.quantile(samples, [0.025, 0.975])
    print(
        f"{metric}: estimate={vals.mean():.4f}, "
        f"95% CI=[{low:.4f}, {high:.4f}], "
        f"documents={len(vals)}"
    )

paired.to_csv(outdir / "section_margin_paired.csv", index=False)
print("\nSaved results to:", outdir)
