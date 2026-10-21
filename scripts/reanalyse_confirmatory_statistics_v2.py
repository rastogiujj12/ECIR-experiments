#!/usr/bin/env python3
"""
Reanalyse the planned contextual-scope contrasts on each dataset's saved evaluation
partition, using the project's paired document bootstrap and Holm adjustment.

Default evaluation partitions, verified against the saved result artifacts:
    QASPER          -> test
    QASA            -> test
    Doc2Dial        -> validation
    MultiDoc2Dial   -> validation

Run from the repository root:
    python3 /path/to/reanalyse_confirmatory_statistics_v2.py --root .

Reads saved query_metrics.csv files only; it does not rerun retrieval or load models.

Primary metric: ndcg_at_5
Primary analysis set: cross_model_core
Contrasts (SCLC minus reference):
    section_constrained vs section_isolated
    section_constrained vs length_matched
    section_constrained vs global

The Holm family is separately defined per dataset and includes every available
model × retrieval-unit-size × planned contrast test (nominally 18 per dataset).
BM25 and fixed_dense are omitted from these inferential contrasts because their
retrieval units differ from the section-bounded targets.
Cross-dataset summaries are descriptive only and include no pooled p-value.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATASETS = ("qasper", "qasa", "doc2dial", "multidoc2dial")
SIZES = (128, 256, 512)
MODELS = ("granite", "jina")
PRIMARY_METRIC = "ndcg_at_5"
PRIMARY_ANALYSIS_SET = "cross_model_core"
DEFAULT_SPLIT_MAP = {
    "qasper": "test",
    "qasa": "test",
    "doc2dial": "validation",
    "multidoc2dial": "validation",
}
SCLC = "section_constrained"
REFERENCES = ("section_isolated", "length_matched", "global")
JOIN_KEYS = ("query_id", "document_id", "analysis_set", "split")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."),
                        help="Repository root (default: current directory)")
    parser.add_argument("--iterations", type=int, default=10000,
                        help="Paired document bootstrap iterations (default: 10000)")
    parser.add_argument("--confidence-level", type=float, default=0.95,
                        help="Confidence level (default: 0.95)")
    parser.add_argument("--seed", type=int, default=20261012,
                        help="Base random seed (default: 20261012)")
    parser.add_argument(
        "--split-map",
        default="qasper=test,qasa=test,doc2dial=validation,multidoc2dial=validation",
        help=("Comma-separated dataset=split mappings. Default: "
              "qasper=test,qasa=test,doc2dial=validation,multidoc2dial=validation"),
    )
    parser.add_argument("--analysis-set", default=PRIMARY_ANALYSIS_SET,
                        help=f"Analysis set (default: {PRIMARY_ANALYSIS_SET})")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("outputs/audit/confirmatory_statistics_dataset_splits"),
                        help="Output directory relative to --root unless absolute")
    return parser.parse_args()


def parse_split_map(value: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"Invalid --split-map item {item!r}; expected dataset=split.")
        dataset, split = (part.strip() for part in item.split("=", 1))
        if not dataset or not split:
            raise SystemExit(f"Invalid --split-map item {item!r}; expected dataset=split.")
        mapping[dataset.casefold()] = split
    missing = [dataset for dataset in DATASETS if dataset not in mapping]
    if missing:
        raise SystemExit(f"--split-map is missing dataset(s): {', '.join(missing)}")
    return {dataset: mapping[dataset] for dataset in DATASETS}


def count_values(series: pd.Series) -> str:
    counts = series.astype("string").fillna("<NA>").value_counts(dropna=False).sort_index()
    return ";".join(f"{key}:{int(value)}" for key, value in counts.items())


def load_metrics(
    path: Path, model: str, dataset: str, size: int, condition: str,
    split: str, analysis_set: str, coverage_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    base_record: dict[str, Any] = {
        "dataset": dataset, "chunk_size_tokens": size, "model": model,
        "condition": condition, "path": str(path), "requested_split": split,
        "requested_analysis_set": analysis_set,
    }
    if not path.exists():
        coverage_rows.append({
            **base_record, "status": "missing_file",
            "rows_before_filter": 0, "filtered_rows": 0, "documents": 0,
            "available_split_counts": "", "available_analysis_set_counts": "",
        })
        return pd.DataFrame(columns=list(JOIN_KEYS) + [PRIMARY_METRIC])

    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    required = set(JOIN_KEYS) | {PRIMARY_METRIC}
    absent = sorted(required - set(frame.columns))
    if absent:
        raise ValueError(f"{path} is missing required columns: {absent}")

    for col in JOIN_KEYS:
        frame[col] = frame[col].astype("string").str.strip()
    available_splits = count_values(frame["split"])
    available_analysis_sets = count_values(frame["analysis_set"])
    frame[PRIMARY_METRIC] = pd.to_numeric(frame[PRIMARY_METRIC], errors="coerce")

    before = len(frame)
    frame = frame[
        (frame["split"].str.casefold() == split.casefold())
        & (frame["analysis_set"].str.casefold() == analysis_set.casefold())
    ].copy()

    if frame[list(JOIN_KEYS)].eq("").any().any():
        raise ValueError(f"{path}: blank identifiers in selected rows")
    duplicates = frame.duplicated(list(JOIN_KEYS), keep=False)
    if duplicates.any():
        sample = frame.loc[duplicates, list(JOIN_KEYS)].head(5).to_dict("records")
        raise ValueError(
            f"{path}: duplicate query keys after filtering; examples: {sample}"
        )

    coverage_rows.append({
        **base_record,
        "status": "ok" if len(frame) else "no_rows_for_requested_split_and_set",
        "rows_before_filter": before, "filtered_rows": len(frame),
        "documents": frame["document_id"].nunique(),
        "available_split_counts": available_splits,
        "available_analysis_set_counts": available_analysis_sets,
    })
    return frame[list(JOIN_KEYS) + [PRIMARY_METRIC]]


def pair_contrast(
    root: Path, dataset: str, size: int, model: str, reference: str,
    split: str, analysis_set: str, coverage_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    base = root / "outputs" / dataset / "evaluation" / f"chunk_{size}"
    sclc_path = base / SCLC / model / "query_metrics.csv"
    ref_path = base / reference / model / "query_metrics.csv"

    a = load_metrics(sclc_path, model, dataset, size, SCLC, split, analysis_set,
                     coverage_rows)
    b = load_metrics(ref_path, model, dataset, size, reference, split, analysis_set,
                     coverage_rows)

    meta: dict[str, Any] = {
        "dataset": dataset, "chunk_size_tokens": size, "model": model,
        "reference_condition": reference, "split": split,
        "analysis_set": analysis_set,
        "sclc_rows_selected": len(a), "reference_rows_selected": len(b),
        "matched_queries": 0, "matched_documents": 0,
        "sclc_only_queries": 0, "reference_only_queries": 0,
        "status": "ok",
    }
    if a.empty or b.empty:
        meta["status"] = "empty_selected_subset"
        return pd.DataFrame(), meta

    merged = a.merge(
        b, on=list(JOIN_KEYS), how="outer", suffixes=("_sclc", "_reference"),
        indicator=True, validate="one_to_one",
    )
    meta["matched_key_count_before_metric_filter"] = int(
        (merged["_merge"] == "both").sum()
    )
    meta["sclc_only_queries"] = int((merged["_merge"] == "left_only").sum())
    meta["reference_only_queries"] = int((merged["_merge"] == "right_only").sum())
    paired = merged.loc[merged["_merge"] == "both"].copy()
    paired = paired.rename(columns={
        f"{PRIMARY_METRIC}_sclc": "sclc_score",
        f"{PRIMARY_METRIC}_reference": "reference_score",
    })
    paired["sclc_score"] = pd.to_numeric(paired["sclc_score"], errors="coerce")
    paired["reference_score"] = pd.to_numeric(paired["reference_score"], errors="coerce")
    paired = paired.dropna(subset=["sclc_score", "reference_score"])
    meta["matched_queries"] = len(paired)
    meta["matched_documents"] = int(paired["document_id"].nunique())
    if paired.empty:
        meta["status"] = "no_paired_metric_rows"
    elif meta["sclc_only_queries"] or meta["reference_only_queries"]:
        meta["status"] = "pairing_mismatch"
    return paired, meta


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not (root / "src" / "sclc").is_dir():
        raise SystemExit(
            f"Could not find src/sclc beneath {root}. Run this from the repository "
            "root or pass --root /path/to/repo."
        )
    if args.iterations < 1000:
        raise SystemExit("--iterations must be at least 1000 for this analysis.")
    if not 0.0 < args.confidence_level < 1.0:
        raise SystemExit("--confidence-level must be between 0 and 1.")
    split_map = parse_split_map(args.split_map)

    sys.path.insert(0, str(root / "src"))
    try:
        from sclc.evaluation.statistics import paired_document_bootstrap, holm_adjust
    except Exception as exc:
        raise SystemExit(
            "Could not import the project's statistical helpers. Run inside the "
            f"project environment. Import error: {exc}"
        ) from exc

    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    pairing_rows: list[dict[str, Any]] = []
    seed_offset = 0

    for dataset in DATASETS:
        split = split_map[dataset]
        for model in MODELS:
            for size in SIZES:
                for reference in REFERENCES:
                    paired, pair_meta = pair_contrast(
                        root, dataset, size, model, reference, split,
                        args.analysis_set, coverage_rows,
                    )
                    pairing_rows.append(pair_meta)
                    if paired.empty:
                        continue

                    seed_offset += 1
                    observed, ci_low, ci_high, p_value, _ = paired_document_bootstrap(
                        paired,
                        first_column="sclc_score",
                        second_column="reference_score",
                        iterations=args.iterations,
                        confidence_level=args.confidence_level,
                        seed=args.seed + seed_offset,
                    )
                    record = dict(pair_meta)
                    record.update({
                        "metric": PRIMARY_METRIC,
                        "n_queries": int(len(paired)),
                        "n_documents": int(paired["document_id"].nunique()),
                        "sclc_mean_ndcg_at_5": float(paired["sclc_score"].mean()),
                        "reference_mean_ndcg_at_5": float(paired["reference_score"].mean()),
                        "mean_difference_sclc_minus_reference": float(observed),
                        "ci95_low": float(ci_low),
                        "ci95_high": float(ci_high),
                        "bootstrap_p_value": float(p_value),
                        "significant_by_unadjusted_ci": bool(ci_low > 0 or ci_high < 0),
                        "bootstrap_iterations": args.iterations,
                        "confidence_level": args.confidence_level,
                        "seed": args.seed + seed_offset,
                    })
                    result_rows.append(record)

    pd.DataFrame(coverage_rows).to_csv(out_dir / "input_coverage.csv", index=False)
    pd.DataFrame(pairing_rows).to_csv(out_dir / "pairing_diagnostics.csv", index=False)

    if not result_rows:
        raise SystemExit(
            "No paired evaluation-partition rows were analysed. Review "
            "input_coverage.csv and pairing_diagnostics.csv."
        )

    results = pd.DataFrame(result_rows)
    results["holm_adjusted_p_value"] = np.nan
    results["significant_after_holm_0_05"] = False
    results["holm_family_size"] = 0
    for dataset, indices in results.groupby("dataset", sort=False).groups.items():
        ix = list(indices)
        adjusted = holm_adjust(results.loc[ix, "bootstrap_p_value"].astype(float).tolist())
        results.loc[ix, "holm_adjusted_p_value"] = adjusted
        results.loc[ix, "significant_after_holm_0_05"] = results.loc[
            ix, "holm_adjusted_p_value"
        ] < 0.05
        results.loc[ix, "holm_family_size"] = len(ix)

    aggregate = (
        results.groupby(["model", "chunk_size_tokens", "reference_condition"], as_index=False)
        .agg(
            datasets_present=("dataset", "nunique"),
            equal_dataset_weighted_mean_effect=("mean_difference_sclc_minus_reference", "mean"),
            datasets_sclc_better=("mean_difference_sclc_minus_reference", lambda x: int((x > 0).sum())),
            datasets_reference_better=("mean_difference_sclc_minus_reference", lambda x: int((x < 0).sum())),
            minimum_dataset_effect=("mean_difference_sclc_minus_reference", "min"),
            maximum_dataset_effect=("mean_difference_sclc_minus_reference", "max"),
        )
    )
    aggregate["interpretation_note"] = (
        "Descriptive equal-weighted average across available datasets' configured "
        "evaluation partitions; not a pooled significance test."
    )

    results.sort_values(
        ["dataset", "model", "chunk_size_tokens", "reference_condition"]
    ).to_csv(out_dir / "primary_contrasts_by_eval_partition.csv", index=False)
    aggregate.sort_values(
        ["model", "chunk_size_tokens", "reference_condition"]
    ).to_csv(out_dir / "descriptive_cross_dataset_summary.csv", index=False)

    expected_per_dataset = len(MODELS) * len(SIZES) * len(REFERENCES)
    lines = [
        "DATASET-SPECIFIC EVALUATION-PARTITION STATISTICAL REANALYSIS",
        "=" * 72,
        f"Repository: {root}",
        f"Primary metric: {PRIMARY_METRIC}",
        f"Primary analysis set: {args.analysis_set}",
        f"Bootstrap iterations: {args.iterations}",
        f"Confidence level: {args.confidence_level}",
        f"Primary contrasts analysed: {len(results)} / {len(DATASETS) * expected_per_dataset} expected",
        "",
        "Evaluation partition by dataset:",
    ]
    for dataset in DATASETS:
        sub = results[results["dataset"] == dataset]
        family = int(sub["holm_family_size"].max()) if not sub.empty else 0
        sig = int(sub["significant_after_holm_0_05"].sum()) if not sub.empty else 0
        lines.append(
            f"  {dataset}: split={split_map[dataset]}; {len(sub)}/{expected_per_dataset} tests; "
            f"Holm family size {family}; {sig} Holm-significant at 0.05"
        )
    lines += [
        "",
        "Holm family: per dataset, across all available model × size × planned contrasts.",
        "No comparison against fixed_dense or BM25 is included.",
        "The cross-dataset summary is descriptive only; no pooled p-value is reported.",
        "",
        "Review input_coverage.csv for the available and selected split labels in every",
        "source file; review pairing_diagnostics.csv for missing or unmatched query keys.",
        "Any dataset below 18/18 tests or any pairing_mismatch must be resolved before",
        "using the results in the paper.",
        "",
        "Files written:",
        "  primary_contrasts_by_eval_partition.csv",
        "  input_coverage.csv",
        "  pairing_diagnostics.csv",
        "  descriptive_cross_dataset_summary.csv",
        "  reanalysis_report.txt",
    ]
    (out_dir / "reanalysis_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
