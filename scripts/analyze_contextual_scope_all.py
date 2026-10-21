#!/usr/bin/env python3
"""Run cross-dataset contextual-scope diagnostics from existing SCLC artifacts.

Run from the root of the ECIR project:
    python scripts/analyze_contextual_scope_all.py

If copied to the project root instead, run:
    python analyze_contextual_scope_all.py

Reads existing embeddings, rankings, qrels and per-query evaluation metrics for
QASA and QASPER. It does not load embedding models or rerun retrieval.
Writes CSV diagnostics and a ZIP suitable for uploading back for review.
"""
from __future__ import annotations

import json
import math
import sys
import traceback
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATASETS = ("qasa", "qasper")
SIZES = (128, 256, 512)
MODELS = ("jina", "granite")
SCOPE_CONDITIONS = ("section_isolated", "section_constrained", "global", "length_matched")
DENSE_CONDITIONS = ("fixed_dense", "section_isolated", "section_constrained", "global", "length_matched")
BM25_METRICS = (
    "ndcg_at_5", "recall_at_5", "evidence_paragraph_recall_at_5",
    "complete_evidence_at_5",
    "evidence_paragraph_recall_at_token_budget_1024",
    "complete_evidence_at_token_budget_2048",
)
BOOTSTRAP_REPS = 2000
SEED = 20261009


def find_project_root(start: Path) -> Path:
    """Find project root containing data/ and outputs/."""
    for candidate in [start, *start.parents]:
        if (candidate / "data").is_dir() and (candidate / "outputs").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate project root containing data/ and outputs/. "
        "Run this script from the ECIR/code project directory."
    )


def csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"WARNING: Could not read CSV {path}: {exc}")
        return None


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def bootstrap_cluster_mean_ci(
    frame: pd.DataFrame,
    value_col: str,
    cluster_col: str = "document_id",
    reps: int = BOOTSTRAP_REPS,
    seed: int = SEED,
) -> tuple[float, float, float, int]:
    """Percentile CI for a mean, resampling document clusters.

    Each bootstrap sample resamples documents with replacement and keeps all
    observations from a sampled document together. The point estimate is the
    pooled observation mean; this avoids treating questions/sections as
    independent bootstrap units.
    """
    work = frame[[cluster_col, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[cluster_col, value_col])
    if work.empty:
        return np.nan, np.nan, np.nan, 0

    grouped = work.groupby(cluster_col)[value_col].agg(["sum", "count"])
    cluster_sums = grouped["sum"].to_numpy(dtype=float)
    cluster_counts = grouped["count"].to_numpy(dtype=float)
    n_clusters = len(grouped)
    point = float(work[value_col].mean())
    if n_clusters < 2:
        return point, np.nan, np.nan, n_clusters

    # Vectorized cluster bootstrap: draw papers with replacement, then pool
    # all observations within each sampled paper. This is much faster for the
    # many dataset/model/size strata than iterating over each sampled paper.
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, n_clusters, size=(reps, n_clusters))
    totals = cluster_sums[sampled].sum(axis=1)
    counts = cluster_counts[sampled].sum(axis=1)
    estimates = totals / np.maximum(counts, 1.0)
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return point, float(lo), float(hi), n_clusters


def bootstrap_paired_document_diff(
    frame: pd.DataFrame,
    value_col: str,
    document_col: str = "document_id",
    reps: int = BOOTSTRAP_REPS,
    seed: int = SEED,
) -> tuple[float, float, float, int]:
    """Paired CI using document-level means of already-paired differences."""
    work = frame[[document_col, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[document_col, value_col])
    doc_means = work.groupby(document_col)[value_col].mean().to_numpy(dtype=float)
    n = len(doc_means)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    point = float(np.mean(doc_means))
    if n == 1:
        return point, np.nan, np.nan, 1
    rng = np.random.default_rng(seed)
    samples = rng.choice(doc_means, size=(reps, n), replace=True).mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return point, float(lo), float(hi), n


def safe_mean(values) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    return float(arr.mean()) if len(arr) else np.nan


def find_enc_dir(root: Path, dataset: str, size: int, condition: str, model: str) -> Path | None:
    base = root / "outputs" / dataset / "encodings" / f"chunk_{size}" / condition / model
    if (base / "documents").is_dir():
        return base / "documents"
    if base.is_dir() and list(base.glob("document_*.npz")):
        return base
    return None


def cosine_diagnostics(root: Path, outdir: Path, notes: list[str]) -> dict[str, pd.DataFrame]:
    """Summarise embedding similarity by document and section for all artifacts."""
    doc_rows: list[dict[str, Any]] = []
    section_rows: list[dict[str, Any]] = []
    sibling_doc_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []

    for dataset in DATASETS:
        for size in SIZES:
            unit_path = root / "data" / dataset / "retrieval_units" / f"chunk_{size}" / "retrieval_units.csv"
            units = csv_if_exists(unit_path)
            if units is None or "retrieval_unit_id" not in units.columns:
                notes.append(f"Cosine skipped: missing retrieval-unit metadata for {dataset} size {size}: {unit_path}")
                continue
            # IDs from the section-bounded conditions map onto these rows. Preserve last non-null section ID if duplicates exist.
            units = units.drop_duplicates("retrieval_unit_id", keep="first")
            metadata: dict[str, dict[str, Any]] = {}
            for row in units.to_dict("records"):
                rid = str(row.get("retrieval_unit_id", ""))
                sec = row.get("parent_section_id")
                sec = None if pd.isna(sec) else str(sec)
                metadata[rid] = {
                    "document_id": str(row.get("document_id", "")),
                    "parent_section_id": sec,
                    "analysis_set": row.get("analysis_set", None),
                }

            for model in MODELS:
                for condition in SCOPE_CONDITIONS:
                    enc_dir = find_enc_dir(root, dataset, size, condition, model)
                    if enc_dir is None:
                        notes.append(f"Cosine unavailable: {dataset}/{size}/{condition}/{model} embedding directory not found")
                        continue
                    archives = sorted(enc_dir.glob("document_*.npz"))
                    if not archives:
                        notes.append(f"Cosine unavailable: no NPZ document files in {enc_dir}")
                        continue
                    manifest_path = enc_dir.parent / "manifest.json"
                    if not manifest_path.is_file():
                        # `documents` is a child of encoding root; check parent too.
                        manifest_path = enc_dir.parent.parent / "manifest.json"
                    if manifest_path.is_file():
                        try:
                            m = json.loads(manifest_path.read_text(encoding="utf-8"))
                            manifests.append({
                                "dataset": dataset, "chunk_size": size,
                                "condition": condition, "model": model,
                                "document_count_manifest": m.get("document_count"),
                                "unit_count_manifest": m.get("unit_count"),
                                "query_count_manifest": m.get("query_count"),
                                "embedding_dimension": m.get("embedding_dimension"),
                            })
                        except Exception as exc:
                            notes.append(f"Manifest read warning {manifest_path}: {exc}")

                    for archive in archives:
                        try:
                            with np.load(archive, allow_pickle=False) as data:
                                vectors = np.asarray(data["embeddings"], dtype=np.float64)
                                unit_ids = data["retrieval_unit_ids"].astype(str)
                        except Exception as exc:
                            notes.append(f"Failed reading {archive}: {exc}")
                            continue
                        if len(unit_ids) != len(vectors):
                            notes.append(f"Embedding/ID length mismatch: {archive}")
                            continue
                        # Normalize defensively so dot products are cosine similarities.
                        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                        vectors = vectors / np.maximum(norms, 1e-12)
                        sim = vectors @ vectors.T
                        infos = [metadata.get(uid) for uid in unit_ids]
                        keep = [i for i, info in enumerate(infos)
                                if info is not None and info.get("parent_section_id") is not None]
                        missing = len(unit_ids) - len(keep)
                        if missing:
                            notes.append(f"{dataset}/{size}/{condition}/{model}: {missing} units in {archive.name} lack section metadata")
                        if len(keep) < 2:
                            continue
                        vectors = vectors[keep]
                        unit_ids = unit_ids[keep]
                        infos = [infos[i] for i in keep]
                        sim = sim[np.ix_(keep, keep)]
                        doc_ids = {info["document_id"] for info in infos}
                        document_id = next(iter(doc_ids)) if len(doc_ids) == 1 else str(archive.stem)
                        if len(doc_ids) != 1:
                            notes.append(f"Multiple document IDs in embedding archive {archive}")
                        sections = [info["parent_section_id"] for info in infos]

                        # All unordered distinct pairs, separated by same/different section.
                        upper_i, upper_j = np.triu_indices(len(unit_ids), k=1)
                        rel_values: dict[str, list[float]] = {"within_section": [], "across_sections": []}
                        for i, j in zip(upper_i, upper_j):
                            relation = "within_section" if sections[i] == sections[j] else "across_sections"
                            rel_values[relation].append(float(sim[i, j]))
                        for relation, vals in rel_values.items():
                            if not vals:
                                continue
                            arr = np.asarray(vals, dtype=float)
                            doc_rows.append({
                                "dataset": dataset, "chunk_size": size,
                                "condition": condition, "model": model,
                                "document_id": document_id, "relation": relation,
                                "chunk_count": len(unit_ids), "pair_count": len(arr),
                                "mean_pair_cosine": float(arr.mean()),
                                "median_pair_cosine": float(np.median(arr)),
                                "p10_pair_cosine": float(np.quantile(arr, .10)),
                                "p90_pair_cosine": float(np.quantile(arr, .90)),
                            })

                        # Section-level average within-section pair similarity, equal weight by section downstream.
                        idx_by_section: dict[str, list[int]] = defaultdict(list)
                        for i, section in enumerate(sections):
                            idx_by_section[section].append(i)
                        for section, idxs in idx_by_section.items():
                            if len(idxs) < 2:
                                continue
                            mat = sim[np.ix_(idxs, idxs)]
                            ii, jj = np.triu_indices(len(idxs), k=1)
                            vals = mat[ii, jj]
                            section_rows.append({
                                "dataset": dataset, "chunk_size": size,
                                "condition": condition, "model": model,
                                "document_id": document_id,
                                "parent_section_id": section,
                                "chunk_count": len(idxs), "pair_count": len(vals),
                                "mean_pair_cosine": float(vals.mean()),
                                "median_pair_cosine": float(np.median(vals)),
                            })
                            # Per-section aggregates of each chunk's sibling similarity.
                            mean_sib = []
                            nearest_sib = []
                            for local_pos in range(len(idxs)):
                                neigh = np.delete(mat[local_pos], local_pos)
                                mean_sib.append(float(np.mean(neigh)))
                                nearest_sib.append(float(np.max(neigh)))
                            sibling_doc_rows.append({
                                "dataset": dataset, "chunk_size": size,
                                "condition": condition, "model": model,
                                "document_id": document_id,
                                "parent_section_id": section,
                                "chunk_count": len(idxs),
                                "mean_sibling_cosine": float(np.mean(mean_sib)),
                                "mean_nearest_sibling_cosine": float(np.mean(nearest_sib)),
                            })

    docs = pd.DataFrame(doc_rows)
    sections_df = pd.DataFrame(section_rows)
    siblings = pd.DataFrame(sibling_doc_rows)
    manifest_df = pd.DataFrame(manifests)

    if not docs.empty:
        # Equal-document-weighted mean vs pair-weighted mean. Bootstrap is over papers.
        summary_rows = []
        groupcols = ["dataset", "chunk_size", "condition", "model", "relation"]
        for keys, group in docs.groupby(groupcols, dropna=False):
            dataset, size, condition, model, relation = keys
            values = group["mean_pair_cosine"].to_numpy(dtype=float)
            row = {
                "dataset": dataset, "chunk_size": size, "condition": condition,
                "model": model, "relation": relation,
                "documents": int(group["document_id"].nunique()),
                "total_pairs": int(group["pair_count"].sum()),
                "equal_document_mean_cosine": float(np.mean(values)),
                "median_document_mean_cosine": float(np.median(values)),
                "pair_weighted_mean_cosine": float(np.average(values, weights=group["pair_count"])),
                "mean_document_p10": float(group["p10_pair_cosine"].mean()),
                "mean_document_p90": float(group["p90_pair_cosine"].mean()),
            }
            row["bootstrap_ci_low"] = np.nan
            row["bootstrap_ci_high"] = np.nan
            row["bootstrap_documents"] = row["documents"]
            if condition in ("section_isolated", "section_constrained"):
                row["bootstrap_ci_low"] = np.nan
                row["bootstrap_ci_high"] = np.nan
            summary_rows.append(row)
        cosine_summary = pd.DataFrame(summary_rows)
    else:
        cosine_summary = pd.DataFrame()

    # Paired document-level SCLC minus section-isolated cosine changes.
    cosine_diff_rows = []
    if not docs.empty:
        for (dataset, size, model, relation), g in docs[docs["condition"].isin(["section_isolated", "section_constrained"])].groupby(
            ["dataset", "chunk_size", "model", "relation"]
        ):
            wide = g.pivot_table(index="document_id", columns="condition", values="mean_pair_cosine", aggfunc="first")
            if not {"section_isolated", "section_constrained"}.issubset(wide.columns):
                continue
            paired = wide.dropna(subset=["section_isolated", "section_constrained"]).copy()
            paired["difference_sclc_minus_isolated"] = paired["section_constrained"] - paired["section_isolated"]
            est, lo, hi, n_docs = bootstrap_paired_document_diff(
                paired.reset_index(), "difference_sclc_minus_isolated", reps=BOOTSTRAP_REPS,
                seed=SEED + int(size) + len(str(dataset)) + len(str(model))
            )
            cosine_diff_rows.append({
                "dataset": dataset, "chunk_size": size, "model": model, "relation": relation,
                "paired_documents": n_docs, "sclc_mean": paired["section_constrained"].mean(),
                "section_isolated_mean": paired["section_isolated"].mean(),
                "difference_sclc_minus_isolated": est, "ci95_low": lo, "ci95_high": hi,
            })

    if not docs.empty:
        docs.to_csv(outdir / "cosine_by_document.csv", index=False)
    if not sections_df.empty:
        sections_df.to_csv(outdir / "cosine_by_section.csv", index=False)
    if not siblings.empty:
        siblings.to_csv(outdir / "sibling_similarity_by_section.csv", index=False)
        sib_summary = (
            siblings.groupby(["dataset", "chunk_size", "condition", "model"])
            .agg(
                documents=("document_id", "nunique"), sections=("parent_section_id", "count"),
                mean_sibling_cosine=("mean_sibling_cosine", "mean"),
                median_section_mean_sibling_cosine=("mean_sibling_cosine", "median"),
                mean_nearest_sibling_cosine=("mean_nearest_sibling_cosine", "mean"),
            ).reset_index()
        )
        sib_summary.to_csv(outdir / "sibling_similarity_summary.csv", index=False)
    else:
        sib_summary = pd.DataFrame()
    if not cosine_summary.empty:
        cosine_summary.to_csv(outdir / "cosine_summary.csv", index=False)
    cosine_diff = pd.DataFrame(cosine_diff_rows)
    if not cosine_diff.empty:
        cosine_diff.to_csv(outdir / "cosine_sclc_vs_isolated_paired.csv", index=False)
    if not manifest_df.empty:
        manifest_df.to_csv(outdir / "encoding_manifest_inventory.csv", index=False)

    return {
        "cosine_by_document": docs,
        "cosine_by_section": sections_df,
        "sibling_similarity_by_section": siblings,
        "sibling_similarity_summary": sib_summary,
        "cosine_summary": cosine_summary,
        "cosine_sclc_vs_isolated_paired": cosine_diff,
        "encoding_manifest_inventory": manifest_df,
    }


def ranking_file(root: Path, dataset: str, size: int, condition: str, model: str) -> Path | None:
    path = root / "outputs" / dataset / "rankings" / f"chunk_{size}" / condition / model / "rankings.jsonl"
    return path if path.is_file() else None


def confusion_diagnostics(root: Path, outdir: Path, notes: list[str]) -> dict[str, pd.DataFrame]:
    detail_rows: list[dict[str, Any]] = []

    for dataset in DATASETS:
        for size in SIZES:
            qrels_path = root / "data" / dataset / "retrieval_units" / f"chunk_{size}" / "qrels.csv"
            qrels = csv_if_exists(qrels_path)
            if qrels is None or not {"query_id", "retrieval_unit_id"}.issubset(qrels.columns):
                notes.append(f"Confusion skipped: missing/incompatible qrels {qrels_path}")
                continue
            if "segmentation_plan" in qrels.columns:
                qrels = qrels[qrels["segmentation_plan"].astype(str).eq("section_bounded")]
            if "relevance" in qrels.columns:
                rel = pd.to_numeric(qrels["relevance"], errors="coerce").fillna(0)
                qrels = qrels.loc[rel > 0]
            qrels["query_id"] = qrels["query_id"].astype(str)
            qrels["retrieval_unit_id"] = qrels["retrieval_unit_id"].astype(str)
            gold_by_query = qrels.groupby("query_id")["retrieval_unit_id"].apply(lambda s: set(s.astype(str))).to_dict()

            for model in MODELS:
                paths = {
                    condition: ranking_file(root, dataset, size, condition, model)
                    for condition in ("section_isolated", "section_constrained")
                }
                if any(p is None for p in paths.values()):
                    notes.append(f"Confusion unavailable: rankings missing for {dataset}/{size}/{model}")
                    continue
                rank_tables: dict[str, pd.DataFrame] = {}
                query_maps: dict[str, dict[str, pd.DataFrame]] = {}
                for condition, path in paths.items():
                    try:
                        ranks = pd.read_json(path, lines=True)
                    except Exception as exc:
                        notes.append(f"Could not read rankings {path}: {exc}")
                        ranks = pd.DataFrame()
                    required = {"query_id", "document_id", "retrieval_unit_id", "rank", "score", "parent_section_id"}
                    if ranks.empty or not required.issubset(ranks.columns):
                        notes.append(f"Incompatible ranking schema: {path}")
                        query_maps[condition] = {}
                        continue
                    ranks["rank"] = pd.to_numeric(ranks["rank"], errors="coerce")
                    ranks["score"] = pd.to_numeric(ranks["score"], errors="coerce")
                    ranks["retrieval_unit_id"] = ranks["retrieval_unit_id"].astype(str)
                    query_maps[condition] = {str(q): g for q, g in ranks.groupby("query_id", sort=False)}
                    rank_tables[condition] = ranks

                common_queries = set(gold_by_query)
                for condition in paths:
                    common_queries &= set(query_maps.get(condition, {}))
                for qid in sorted(common_queries):
                    relevant = gold_by_query.get(qid, set())
                    if not relevant:
                        continue
                    for condition in paths:
                        g = query_maps[condition][qid].sort_values("rank")
                        # Only sections that contain a gold section-bounded unit for this query.
                        gold_rows_all = g[g["retrieval_unit_id"].isin(relevant)]
                        if gold_rows_all.empty:
                            continue
                        for section in gold_rows_all["parent_section_id"].dropna().astype(str).unique():
                            sg = g[g["parent_section_id"].astype(str).eq(section)]
                            gold_section = sg[sg["retrieval_unit_id"].isin(relevant)]
                            wrong_section = sg[~sg["retrieval_unit_id"].isin(relevant)]
                            if gold_section.empty:
                                continue
                            best_gold_rank = float(gold_section["rank"].min())
                            best_gold_score = float(gold_section["score"].max())
                            if wrong_section.empty:
                                best_wrong_rank = np.nan
                                best_wrong_score = np.nan
                                wrong_before = False
                            else:
                                best_wrong_rank = float(wrong_section["rank"].min())
                                best_wrong_score = float(wrong_section["score"].max())
                                wrong_before = best_wrong_rank < best_gold_rank
                            detail_rows.append({
                                "dataset": dataset, "chunk_size": size, "model": model,
                                "condition": condition, "query_id": str(qid),
                                "document_id": str(g["document_id"].iloc[0]),
                                "parent_section_id": section,
                                "best_gold_rank": best_gold_rank,
                                "best_gold_score": best_gold_score,
                                "best_same_section_distractor_rank": best_wrong_rank,
                                "best_same_section_distractor_score": best_wrong_score,
                                "gold_minus_distractor_margin": (
                                    best_gold_score - best_wrong_score
                                    if np.isfinite(best_wrong_score) else np.nan
                                ),
                                "gold_hit_at_5_within_section": best_gold_rank <= 5,
                                "distractor_before_gold": bool(wrong_before),
                                "same_section_distractors_at_5": int(
                                    ((sg["rank"] <= 5) & (~sg["retrieval_unit_id"].isin(relevant))).sum()
                                ),
                            })

    details = pd.DataFrame(detail_rows)
    if details.empty:
        notes.append("No retrieval-confusion rows were created; check qrels/ranking schemas and unit IDs.")
        return {"retrieval_confusion_by_query_section": details}

    details.to_csv(outdir / "retrieval_confusion_by_query_section.csv", index=False)
    grp = ["dataset", "chunk_size", "model", "condition"]
    summary = (
        details.groupby(grp)
        .agg(
            query_section_pairs=("query_id", "size"),
            queries=("query_id", "nunique"),
            documents=("document_id", "nunique"),
            gold_hit_at_5=("gold_hit_at_5_within_section", "mean"),
            distractor_before_gold_rate=("distractor_before_gold", "mean"),
            mean_gold_minus_distractor_margin=("gold_minus_distractor_margin", "mean"),
            median_gold_minus_distractor_margin=("gold_minus_distractor_margin", "median"),
            mean_same_section_distractors_at_5=("same_section_distractors_at_5", "mean"),
        ).reset_index()
    )
    summary.to_csv(outdir / "retrieval_confusion_summary.csv", index=False)

    # Strict paired query-section comparisons between SCLC and section-isolated.
    paired_rows = []
    compare_metrics = [
        "gold_hit_at_5_within_section", "distractor_before_gold",
        "gold_minus_distractor_margin", "same_section_distractors_at_5",
    ]
    keys = ["dataset", "chunk_size", "model", "query_id", "document_id", "parent_section_id"]
    for (dataset, size, model), block in details.groupby(["dataset", "chunk_size", "model"]):
        use = block[block["condition"].isin(["section_isolated", "section_constrained"])]
        wide = use.pivot_table(index=keys[3:], columns="condition", values=compare_metrics, aggfunc="first")
        if wide.empty:
            continue
        # Flatten MultiIndex metric/condition columns.
        wide.columns = [f"{metric}__{condition}" for metric, condition in wide.columns]
        wide = wide.reset_index()
        needed = []
        for metric in compare_metrics:
            for condition in ("section_isolated", "section_constrained"):
                col = f"{metric}__{condition}"
                if col not in wide.columns:
                    wide[col] = np.nan
                needed.append(col)
        wide = wide.dropna(subset=needed)
        if wide.empty:
            continue
        wide["dataset"] = dataset
        wide["chunk_size"] = size
        wide["model"] = model
        wide["diff_gold_hit_sclc_minus_isolated"] = (
            wide["gold_hit_at_5_within_section__section_constrained"].astype(float)
            - wide["gold_hit_at_5_within_section__section_isolated"].astype(float)
        )
        wide["diff_distractor_rate_sclc_minus_isolated"] = (
            wide["distractor_before_gold__section_constrained"].astype(float)
            - wide["distractor_before_gold__section_isolated"].astype(float)
        )
        wide["diff_margin_sclc_minus_isolated"] = (
            wide["gold_minus_distractor_margin__section_constrained"].astype(float)
            - wide["gold_minus_distractor_margin__section_isolated"].astype(float)
        )
        wide["diff_distractors_at5_sclc_minus_isolated"] = (
            wide["same_section_distractors_at_5__section_constrained"].astype(float)
            - wide["same_section_distractors_at_5__section_isolated"].astype(float)
        )
        paired_rows.extend(wide.to_dict("records"))
    paired = pd.DataFrame(paired_rows)
    if not paired.empty:
        paired.to_csv(outdir / "retrieval_confusion_paired.csv", index=False)

        boot_rows = []
        diffs = [
            ("gold_hit_at_5", "diff_gold_hit_sclc_minus_isolated"),
            ("distractor_before_gold", "diff_distractor_rate_sclc_minus_isolated"),
            ("gold_minus_distractor_margin", "diff_margin_sclc_minus_isolated"),
            ("same_section_distractors_at_5", "diff_distractors_at5_sclc_minus_isolated"),
        ]
        for (dataset, size, model), block in paired.groupby(["dataset", "chunk_size", "model"]):
            for label, value_col in diffs:
                est, lo, hi, ndoc = bootstrap_paired_document_diff(
                    block, value_col, reps=BOOTSTRAP_REPS,
                    seed=SEED + int(size) + len(dataset) + len(model) + len(label)
                )
                boot_rows.append({
                    "dataset": dataset, "chunk_size": size, "model": model,
                    "metric": label, "paired_query_sections": len(block),
                    "documents": ndoc, "sclc_minus_isolated": est,
                    "ci95_low": lo, "ci95_high": hi,
                })
        boot = pd.DataFrame(boot_rows)
        boot.to_csv(outdir / "retrieval_confusion_paired_bootstrap.csv", index=False)
    else:
        boot = pd.DataFrame()

    return {
        "retrieval_confusion_by_query_section": details,
        "retrieval_confusion_summary": summary,
        "retrieval_confusion_paired": paired,
        "retrieval_confusion_paired_bootstrap": boot,
    }


def load_metric_table(root: Path, dataset: str, size: int, condition: str, model: str | None) -> pd.DataFrame | None:
    if condition == "bm25":
        path = root / "outputs" / dataset / "evaluation" / f"chunk_{size}" / "bm25" / "query_metrics.csv"
    else:
        if model is None:
            return None
        path = root / "outputs" / dataset / "evaluation" / f"chunk_{size}" / condition / model / "query_metrics.csv"
    table = csv_if_exists(path)
    if table is None or "query_id" not in table.columns:
        return None
    table["query_id"] = table["query_id"].astype(str)
    return table


def bm25_comparison(root: Path, outdir: Path, notes: list[str]) -> dict[str, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []
    for dataset in DATASETS:
        for size in SIZES:
            bm25 = load_metric_table(root, dataset, size, "bm25", None)
            if bm25 is None:
                notes.append(f"BM25 comparison unavailable: missing BM25 per-query metrics for {dataset}/{size}")
                continue
            for model in MODELS:
                for condition in DENSE_CONDITIONS:
                    dense = load_metric_table(root, dataset, size, condition, model)
                    if dense is None:
                        notes.append(f"BM25 comparison unavailable: missing metrics for {dataset}/{size}/{condition}/{model}")
                        continue
                    keys = ["query_id"]
                    for col in ("analysis_set", "split"):
                        if col in bm25.columns and col in dense.columns:
                            keys.append(col)
                    metrics = [m for m in BM25_METRICS if m in bm25.columns and m in dense.columns]
                    if not metrics:
                        notes.append(f"No shared metrics for BM25 vs {condition}: {dataset}/{size}/{model}")
                        continue
                    left_cols = list(dict.fromkeys(keys + ["document_id"] if "document_id" in bm25.columns else keys)) + metrics
                    right_cols = list(dict.fromkeys(keys + ["document_id"] if "document_id" in dense.columns else keys)) + metrics
                    a = bm25[left_cols].copy()
                    b = dense[right_cols].copy()
                    # Avoid duplicate-key cartesian merges if an evaluation file unexpectedly repeats a query.
                    if a.duplicated(keys).any():
                        notes.append(f"BM25 metrics duplicate query keys; keeping first for {dataset}/{size}")
                        a = a.drop_duplicates(keys, keep="first")
                    if b.duplicated(keys).any():
                        notes.append(f"Dense metrics duplicate query keys; keeping first for {dataset}/{size}/{condition}/{model}")
                        b = b.drop_duplicates(keys, keep="first")
                    merged = a.merge(b, on=keys, how="inner", suffixes=("_bm25", "_dense"))
                    if merged.empty:
                        notes.append(f"No paired BM25/dense query metrics for {dataset}/{size}/{condition}/{model}")
                        continue
                    doc_col = "document_id_bm25" if "document_id_bm25" in merged.columns else (
                        "document_id_dense" if "document_id_dense" in merged.columns else None
                    )
                    if doc_col is None:
                        # Query IDs in these data are paper-prefixed; use that prefix only as a fallback cluster ID.
                        merged["document_id_cluster"] = merged["query_id"].astype(str).str.split(":").str[0]
                        doc_col = "document_id_cluster"
                    for metric in metrics:
                        bm_col = f"{metric}_bm25"
                        dense_col = f"{metric}_dense"
                        merged[bm_col] = pd.to_numeric(merged[bm_col], errors="coerce")
                        merged[dense_col] = pd.to_numeric(merged[dense_col], errors="coerce")
                        merged[f"dense_minus_bm25__{metric}"] = merged[dense_col] - merged[bm_col]
                        valid = merged.dropna(subset=[bm_col, dense_col]).copy()
                        if valid.empty:
                            continue
                        grp_cols = [c for c in ["analysis_set", "split"] if c in valid.columns]
                        groups = valid.groupby(grp_cols, dropna=False, sort=False) if grp_cols else [((), valid)]
                        for grp_key, g in groups:
                            if not isinstance(grp_key, tuple):
                                grp_key = (grp_key,)
                            group_vals = dict(zip(grp_cols, grp_key))
                            d = g[[doc_col, f"dense_minus_bm25__{metric}"]].rename(columns={doc_col: "document_id", f"dense_minus_bm25__{metric}": "difference"})
                            if metric == "ndcg_at_5":
                                est, lo, hi, ndoc = bootstrap_cluster_mean_ci(
                                    d, "difference", reps=BOOTSTRAP_REPS,
                                    seed=SEED + int(size) + len(dataset) + len(model) + len(condition) + len(metric) + len(str(grp_key))
                                )
                            else:
                                est, lo, hi, ndoc = np.nan, np.nan, np.nan, int(g[doc_col].nunique())
                            diff = g[f"dense_minus_bm25__{metric}"]
                            bm_vals = g[bm_col]
                            dense_vals = g[dense_col]
                            row = {
                                "dataset": dataset, "chunk_size": size, "model": model,
                                "dense_condition": condition, "metric": metric,
                                **group_vals,
                                "queries": int(len(g)), "documents": int(g[doc_col].nunique()),
                                "bm25_mean": float(bm_vals.mean()),
                                "dense_mean": float(dense_vals.mean()),
                                "dense_minus_bm25_query_mean": float(diff.mean()),
                                "dense_minus_bm25_document_cluster_bootstrap_mean": est,
                                "ci95_low": lo, "ci95_high": hi,
                                "bm25_win_rate": float((bm_vals > dense_vals).mean()),
                                "dense_win_rate": float((dense_vals > bm_vals).mean()),
                                "tie_rate": float(np.isclose(bm_vals, dense_vals, atol=1e-12).mean()),
                            }
                            summary_rows.append(row)
                    # Save useful per-query paired differences (without copying all original columns).
                    id_cols = [c for c in keys if c in merged.columns]
                    if doc_col in merged.columns:
                        id_cols.append(doc_col)
                    keep_cols = id_cols + [c for m in metrics for c in [f"{m}_bm25", f"{m}_dense", f"dense_minus_bm25__{m}"] if c in merged.columns]
                    frame = merged[keep_cols].copy()
                    frame["dataset"] = dataset
                    frame["chunk_size"] = size
                    frame["model"] = model
                    frame["dense_condition"] = condition
                    detail_frames.append(frame)

    summary = pd.DataFrame(summary_rows)
    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    if not summary.empty:
        summary.to_csv(outdir / "bm25_vs_dense_summary.csv", index=False)
    if not details.empty:
        details.to_csv(outdir / "bm25_vs_dense_paired_queries.csv", index=False)
    return {"bm25_vs_dense_summary": summary, "bm25_vs_dense_paired_queries": details}


def lexical_overlap_diagnostics(root: Path, outdir: Path, notes: list[str], bm25_details: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Measure question-term overlap with gold evidence chunks and relate it to BM25 wins.

    Uses the section-bounded chunk text and positive qrels available for each size.
    These are descriptive lexical-overlap diagnostics, not causal attribution.
    """
    import re

    stopwords = set("""a an the and or but if then than to of in on at by for from with as is are was were be been being it its this that these those we they he she you your their our can could would should may might do does did how what when where why which who whom into over under between within without about after before during through across any almost just more most less very such not no yes all both each few many other same different only also have has had having using used use paper authors author model models method methods approach result results work works study proposed""".split())
    word_re = re.compile(r"[a-z0-9]+(?:['’][a-z0-9]+)?")

    def tokens(text: Any) -> list[str]:
        if text is None or (isinstance(text, float) and math.isnan(text)):
            return []
        return word_re.findall(str(text).lower())

    pair_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for size in SIZES:
            qpath = root / "data" / dataset / "retrieval_units" / f"chunk_{size}" / "queries.jsonl"
            upath = root / "data" / dataset / "retrieval_units" / f"chunk_{size}" / "section_bounded_units.jsonl"
            rpath = root / "data" / dataset / "retrieval_units" / f"chunk_{size}" / "qrels.csv"
            if not (qpath.is_file() and upath.is_file() and rpath.is_file()):
                notes.append(f"Lexical overlap unavailable: missing query/unit/qrels file for {dataset}/{size}")
                continue
            try:
                qrecs = list(read_jsonl(qpath))
                urecs = list(read_jsonl(upath))
            except Exception as exc:
                notes.append(f"Lexical overlap JSONL read failed for {dataset}/{size}: {exc}")
                continue
            if not qrecs or not urecs:
                notes.append(f"Lexical overlap skipped: empty query/unit records for {dataset}/{size}")
                continue
            qdf = pd.DataFrame(qrecs)
            udf = pd.DataFrame(urecs)
            qid_col = next((c for c in ["query_id", "id"] if c in qdf.columns), None)
            qtext_col = next((c for c in ["question", "query", "query_text", "question_text", "text", "content"] if c in qdf.columns), None)
            uid_col = next((c for c in ["retrieval_unit_id", "unit_id", "id"] if c in udf.columns), None)
            utext_col = next((c for c in ["text", "unit_text", "content", "retrieval_unit_text"] if c in udf.columns), None)
            if not qid_col or not qtext_col or not uid_col or not utext_col:
                notes.append(
                    f"Lexical overlap skipped {dataset}/{size}: couldn't find query/unit identifier/text columns; "
                    f"query columns={list(qdf.columns)}, unit columns={list(udf.columns)}"
                )
                continue
            qdf[qid_col] = qdf[qid_col].astype(str)
            udf[uid_col] = udf[uid_col].astype(str)
            qtext = qdf.drop_duplicates(qid_col).set_index(qid_col)[qtext_col].to_dict()
            unit_text = udf.drop_duplicates(uid_col).set_index(uid_col)[utext_col].to_dict()
            qmeta_cols = [c for c in [qid_col, "document_id", "split", "analysis_set", qtext_col] if c in qdf.columns]
            qmeta = qdf[qmeta_cols].drop_duplicates(qid_col).copy()
            if qid_col != "query_id":
                qmeta = qmeta.rename(columns={qid_col: "query_id"})
            qmeta["query_id"] = qmeta["query_id"].astype(str)
            qmeta = qmeta.rename(columns={qtext_col: "question_text"})
            qrels = pd.read_csv(rpath)
            if not {"query_id", "retrieval_unit_id"}.issubset(qrels.columns):
                notes.append(f"Lexical overlap skipped: qrels schema missing keys at {rpath}")
                continue
            qrels["query_id"] = qrels["query_id"].astype(str)
            qrels["retrieval_unit_id"] = qrels["retrieval_unit_id"].astype(str)
            if "segmentation_plan" in qrels.columns:
                qrels = qrels[qrels["segmentation_plan"].astype(str).eq("section_bounded")]
            if "relevance" in qrels.columns:
                qrels = qrels[pd.to_numeric(qrels["relevance"], errors="coerce").fillna(0) > 0]

            for row in qrels[["query_id", "retrieval_unit_id"]].drop_duplicates().to_dict("records"):
                qid = row["query_id"]
                uid = row["retrieval_unit_id"]
                if qid not in qtext or uid not in unit_text:
                    continue
                question = str(qtext[qid])
                evidence_chunk = str(unit_text[uid])
                q_tokens = set(tokens(question))
                e_tokens = set(tokens(evidence_chunk))
                q_content = q_tokens - stopwords
                e_content = e_tokens - stopwords
                shared_raw = q_tokens & e_tokens
                shared_content = q_content & e_content
                pair_rows.append({
                    "dataset": dataset, "chunk_size": size,
                    "query_id": qid, "retrieval_unit_id": uid,
                    "question_token_count": len(q_tokens),
                    "question_content_token_count": len(q_content),
                    "gold_chunk_content_token_count": len(e_content),
                    "shared_query_terms_raw": len(shared_raw),
                    "shared_query_content_terms": len(shared_content),
                    "query_term_coverage_raw": len(shared_raw) / len(q_tokens) if q_tokens else np.nan,
                    "query_content_term_coverage": len(shared_content) / len(q_content) if q_content else np.nan,
                    "content_term_jaccard": len(shared_content) / len(q_content | e_content) if (q_content | e_content) else np.nan,
                })

            pairs_df = pd.DataFrame([r for r in pair_rows if r["dataset"] == dataset and r["chunk_size"] == size])
            if pairs_df.empty:
                notes.append(f"Lexical overlap found no mappable query/gold-chunk pairs for {dataset}/{size}; check JSONL field names and qrels IDs")
                continue
            meta_small = qmeta.rename(columns={"question_text": "_question_text"})
            pairs_df = pairs_df.merge(meta_small[[c for c in ["query_id", "document_id", "split", "analysis_set"] if c in meta_small.columns]], on="query_id", how="left")
            # Aggregate all gold chunks per question. Max overlap captures the best-aligned supporting chunk.
            agg = pairs_df.groupby(["dataset", "chunk_size", "query_id"], as_index=False).agg(
                gold_chunks_mapped=("retrieval_unit_id", "nunique"),
                max_query_term_coverage_raw=("query_term_coverage_raw", "max"),
                mean_query_term_coverage_raw=("query_term_coverage_raw", "mean"),
                max_query_content_term_coverage=("query_content_term_coverage", "max"),
                mean_query_content_term_coverage=("query_content_term_coverage", "mean"),
                max_content_term_jaccard=("content_term_jaccard", "max"),
                mean_content_term_jaccard=("content_term_jaccard", "mean"),
                mean_shared_query_content_terms=("shared_query_content_terms", "mean"),
                question_content_token_count=("question_content_token_count", "first"),
            )
            agg = agg.merge(meta_small[[c for c in ["query_id", "document_id", "split", "analysis_set"] if c in meta_small.columns]].drop_duplicates("query_id"), on="query_id", how="left")
            # Quartiles are created independently for each dataset × size.
            vals = pd.to_numeric(agg["max_query_content_term_coverage"], errors="coerce")
            valid = vals.notna()
            agg["lexical_overlap_tertile"] = "unknown"
            if valid.sum() >= 3 and vals[valid].nunique() >= 2:
                n_bins = min(3, int(vals[valid].nunique()))
                labels = ["low", "medium", "high"][-n_bins:]
                try:
                    agg.loc[valid, "lexical_overlap_tertile"] = pd.qcut(
                        vals[valid].rank(method="first"), q=n_bins, labels=labels
                    ).astype(str)
                except Exception:
                    agg.loc[valid, "lexical_overlap_tertile"] = "available"
            query_rows.extend(agg.to_dict("records"))
            pairs_df.to_csv(outdir / f"lexical_overlap_gold_chunks_{dataset}_{size}.csv", index=False)

    pairs_all = pd.DataFrame(pair_rows)
    queries_all = pd.DataFrame(query_rows)
    if not pairs_all.empty:
        pairs_all.to_csv(outdir / "lexical_overlap_by_gold_chunk.csv", index=False)
    if not queries_all.empty:
        queries_all.to_csv(outdir / "lexical_overlap_by_query.csv", index=False)

    bins_summary = pd.DataFrame()
    if not queries_all.empty and not bm25_details.empty:
        # Paired query-level ranking outcomes vs gold-chunk lexical overlap.
        detail = bm25_details.merge(
            queries_all[["dataset", "chunk_size", "query_id", "max_query_content_term_coverage", "mean_query_content_term_coverage", "lexical_overlap_tertile"]].drop_duplicates(["dataset", "chunk_size", "query_id"]),
            on=["dataset", "chunk_size", "query_id"], how="inner"
        )
        # Keep nDCG@5 as the primary stratified diagnostic; include evidence-recall outcomes if available.
        metric_cols = [c for c in ["ndcg_at_5_bm25", "ndcg_at_5_dense", "dense_minus_bm25__ndcg_at_5",
                                   "evidence_paragraph_recall_at_5_bm25", "evidence_paragraph_recall_at_5_dense",
                                   "dense_minus_bm25__evidence_paragraph_recall_at_5"] if c in detail.columns]
        group_cols = [c for c in ["dataset", "chunk_size", "model", "dense_condition", "lexical_overlap_tertile"] if c in detail.columns]
        if group_cols and "dense_minus_bm25__ndcg_at_5" in detail.columns:
            bins_summary = detail.groupby(group_cols, dropna=False).agg(
                queries=("query_id", "nunique"),
                bm25_ndcg_at_5=("ndcg_at_5_bm25", "mean"),
                dense_ndcg_at_5=("ndcg_at_5_dense", "mean"),
                dense_minus_bm25_ndcg_at_5=("dense_minus_bm25__ndcg_at_5", "mean"),
                bm25_win_rate_ndcg_at_5=("ndcg_at_5_bm25", lambda x: np.nan),
            ).reset_index()
            # Exact win-rate and evidence-recall summaries where present.
            detail["bm25_wins_ndcg_at_5"] = detail["ndcg_at_5_bm25"] > detail["ndcg_at_5_dense"]
            winrates = detail.groupby(group_cols)["bm25_wins_ndcg_at_5"].mean().rename("bm25_win_rate_ndcg_at_5").reset_index()
            bins_summary = bins_summary.drop(columns=["bm25_win_rate_ndcg_at_5"]).merge(winrates, on=group_cols, how="left")
            if "dense_minus_bm25__evidence_paragraph_recall_at_5" in detail.columns:
                ev = detail.groupby(group_cols).agg(
                    dense_minus_bm25_evidence_recall_at_5=("dense_minus_bm25__evidence_paragraph_recall_at_5", "mean"),
                    bm25_evidence_recall_at_5=("evidence_paragraph_recall_at_5_bm25", "mean"),
                    dense_evidence_recall_at_5=("evidence_paragraph_recall_at_5_dense", "mean"),
                ).reset_index()
                bins_summary = bins_summary.merge(ev, on=group_cols, how="left")
            bins_summary.to_csv(outdir / "bm25_by_lexical_overlap.csv", index=False)
        else:
            notes.append("Lexical-overlap strata not joined to BM25 metrics because expected paired nDCG columns were absent")
    elif queries_all.empty:
        notes.append("Lexical overlap did not produce per-query data; no BM25-by-overlap analysis created")

    return {
        "lexical_overlap_by_gold_chunk": pairs_all,
        "lexical_overlap_by_query": queries_all,
        "bm25_by_lexical_overlap": bins_summary,
    }


def write_readme(outdir: Path, root: Path, result_frames: dict[str, pd.DataFrame], notes: list[str]) -> None:
    lines = [
        "CONTEXTUAL SCOPE DIAGNOSTICS — RUN SUMMARY",
        "",
        f"Project root: {root}",
        "Datasets attempted: QASA, QASPER",
        "Retrieval-unit sizes attempted: 128, 256, 512",
        "Embedding models attempted: Jina, Granite",
        "Conditions: section-isolated, section-constrained (SCLC), global (GLC), length-matched (LMLC); BM25 and fixed-dense baselines for existing evaluation comparisons.",
        "",
        "This script reads existing artifacts only. It does not load models or rerun encoding/retrieval.",
        "Cosine summaries are computed per paper and then aggregated, so papers with many chunk pairs do not solely determine the equal-paper summaries.",
        "Retrieval confusion compares gold evidence units and non-gold chunks within the same parent section.",
        "BM25 comparisons pair per-query evaluation metrics on common query IDs and bootstrap document clusters. Lexical overlap uses token overlap between each question and its gold evidence-containing section-bounded chunks; it is descriptive and may be affected by chunk size.",
        "These are diagnostics; similarity associations do not prove causal mechanisms.",
        "",
        "OUTPUT INVENTORY",
    ]
    for name, frame in sorted(result_frames.items()):
        lines.append(f"- {name}: {len(frame):,} rows")
    lines.append("")
    if notes:
        lines.append(f"WARNINGS / MISSING ARTIFACTS ({len(notes)}):")
        lines.extend(f"- {n}" for n in notes)
    else:
        lines.append("No missing-artifact warnings.")
    (outdir / "README_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    root = find_project_root(Path.cwd().resolve())
    outdir = root / "outputs" / "contextual_scope_diagnostics"
    outdir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    print(f"Project root: {root}")
    print(f"Output directory: {outdir}")
    print("Reading existing artifacts; no models will be loaded.\n")

    results: dict[str, pd.DataFrame] = {}
    print("[1/4] Computing cosine-similarity diagnostics across datasets/models/sizes...")
    try:
        results.update(cosine_diagnostics(root, outdir, notes))
    except Exception as exc:
        notes.append(f"Cosine stage failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()

    print("[2/4] Comparing within-section retrieval confusion for SCLC vs section-isolated...")
    try:
        results.update(confusion_diagnostics(root, outdir, notes))
    except Exception as exc:
        notes.append(f"Retrieval-confusion stage failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()

    print("[3/4] Comparing BM25 with dense conditions using paired query metrics...")
    bm25_results: dict[str, pd.DataFrame] = {}
    try:
        bm25_results = bm25_comparison(root, outdir, notes)
        results.update(bm25_results)
    except Exception as exc:
        notes.append(f"BM25 comparison stage failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()

    print("[4/4] Measuring lexical overlap between questions and gold evidence chunks...")
    try:
        results.update(lexical_overlap_diagnostics(
            root, outdir, notes,
            bm25_results.get("bm25_vs_dense_paired_queries", pd.DataFrame())
        ))
    except Exception as exc:
        notes.append(f"Lexical-overlap stage failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()

    write_readme(outdir, root, results, notes)
    zip_path = root / "outputs" / "contextual_scope_diagnostics_upload.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(outdir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(Path("contextual_scope_diagnostics") / path.relative_to(outdir)))

    print("\n=== COMPLETION ===")
    for name, frame in sorted(results.items()):
        print(f"{name}: {len(frame):,} rows")
    print(f"Warnings: {len(notes)}")
    print(f"Report directory: {outdir}")
    print(f"Upload-ready ZIP: {zip_path}")
    print("Upload contextual_scope_diagnostics_upload.zip back here for review.")
    return 0 if any(not frame.empty for frame in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
