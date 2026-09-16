from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from sclc.config import AppConfig
from sclc.options import EmbeddingModel, RetrievalCondition
from sclc.paths import evaluation_dir

COMPARISONS: tuple[tuple[RetrievalCondition, RetrievalCondition], ...] = (
    # Every signed comparison follows first condition minus second condition.
    (RetrievalCondition.BM25, RetrievalCondition.FIXED_DENSE),
    (RetrievalCondition.SECTION_ISOLATED, RetrievalCondition.FIXED_DENSE),
    (RetrievalCondition.SECTION_CONSTRAINED, RetrievalCondition.SECTION_ISOLATED),
    (RetrievalCondition.SECTION_CONSTRAINED, RetrievalCondition.LENGTH_MATCHED),
    (RetrievalCondition.GLOBAL, RetrievalCondition.SECTION_CONSTRAINED),
)


def _fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _evaluation_dir(
    config: AppConfig,
    condition: RetrievalCondition,
    model: EmbeddingModel | None,
    *,
    chunk_size: int,
) -> Path:
    base = evaluation_dir(config, chunk_size)
    if condition is RetrievalCondition.BM25:
        return base / condition.value
    assert model is not None
    return base / condition.value / model.value


def _load_metrics(
    config: AppConfig,
    condition: RetrievalCondition,
    model: EmbeddingModel | None,
    *,
    chunk_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    directory = _evaluation_dir(config, condition, model, chunk_size=chunk_size)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        label = condition.value if model is None else f"{condition.value}/{model.value}"
        raise FileNotFoundError(f"Evaluation for {label} is missing. Run `sclc evaluate` first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(
        directory / manifest["files"]["query_metrics"],
        dtype={"query_id": "string", "document_id": "string"},
    )
    return frame, manifest


def paired_document_bootstrap(
    frame: pd.DataFrame,
    *,
    first_column: str,
    second_column: str,
    iterations: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float, float, float, np.ndarray]:
    required = {"document_id", first_column, second_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Bootstrap frame is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Cannot bootstrap an empty comparison")

    documents = sorted(frame["document_id"].astype(str).unique())
    if not documents:
        raise ValueError("No documents are available for bootstrapping")
    differences = frame[first_column].to_numpy(dtype=float) - frame[second_column].to_numpy(
        dtype=float
    )
    observed = float(differences.mean())
    row_indices_by_document = {
        document_id: np.flatnonzero(frame["document_id"].astype(str).to_numpy() == document_id)
        for document_id in documents
    }
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=np.float64)
    null_samples = np.empty(iterations, dtype=np.float64)
    centered_differences = differences - observed
    for iteration in range(iterations):
        sampled_documents = rng.choice(documents, size=len(documents), replace=True)
        sampled_indices = np.concatenate(
            [row_indices_by_document[str(document_id)] for document_id in sampled_documents]
        )
        samples[iteration] = float(differences[sampled_indices].mean())
        null_samples[iteration] = float(centered_differences[sampled_indices].mean())

    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    p_value = (
        np.count_nonzero(np.abs(null_samples) >= abs(observed)) + 1.0
    ) / (iterations + 1.0)
    return observed, float(lower), float(upper), float(p_value), samples


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    if not p_values:
        return []
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _analysis_groups(model: EmbeddingModel) -> list[tuple[str, set[str]]]:
    groups = [("cross_model_core", {"cross_model_core"})]
    if model is EmbeddingModel.GRANITE:
        groups.extend(
            [
                ("granite_extended", {"granite_extended"}),
                ("all_eligible", {"cross_model_core", "granite_extended"}),
            ]
        )
    return groups


def _group_slices(frame: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    slices = [("overall", "all", frame)]
    for query_type, group in frame.groupby("query_type", dropna=False):
        if str(query_type) in {"unclassified", "uncertain"}:
            continue
        slices.append(("query_type", str(query_type), group))
    return slices


def compare_conditions(
    config: AppConfig,
    *,
    chunk_size: int,
    models: Sequence[EmbeddingModel] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    selected_models = tuple(
        dict.fromkeys(tuple(EmbeddingModel) if models is None else models)
    )
    if not selected_models:
        raise ValueError("At least one embedding model must be selected")
    unknown_models = set(selected_models).difference(EmbeddingModel)
    if unknown_models:
        raise ValueError(f"Unknown embedding models: {sorted(unknown_models)}")

    output_dir = evaluation_dir(config, chunk_size) / "comparisons"
    if set(selected_models) != set(EmbeddingModel):
        output_dir = output_dir / "__".join(model.value for model in selected_models)
    output_path = output_dir / "pairwise_comparisons.csv"
    manifest_path = output_dir / "manifest.json"
    bootstrap_dir = output_dir / "bootstrap"
    candidates_path = output_dir / "error_analysis_candidates.csv"

    source_manifests: dict[str, str] = {}
    loaded: dict[tuple[RetrievalCondition, EmbeddingModel | None], pd.DataFrame] = {}
    bm25_frame, bm25_manifest = _load_metrics(
        config, RetrievalCondition.BM25, None, chunk_size=chunk_size
    )
    if "split" not in bm25_frame.columns:
        raise ValueError("BM25 query metrics are missing the split column")
    bm25_frame = bm25_frame[
        bm25_frame["split"] == config.evaluation.confirmatory_split
    ].copy()
    if bm25_frame.empty:
        raise RuntimeError("No confirmatory-split BM25 queries are available")
    loaded[(RetrievalCondition.BM25, None)] = bm25_frame
    source_manifests["bm25"] = bm25_manifest["configuration_fingerprint"]
    for model in selected_models:
        for condition in RetrievalCondition:
            if condition is RetrievalCondition.BM25:
                continue
            frame, manifest = _load_metrics(
                config, condition, model, chunk_size=chunk_size
            )
            if "split" not in frame.columns:
                raise ValueError(
                    f"{condition.value}/{model.value} query metrics are missing split"
                )
            frame = frame[
                frame["split"] == config.evaluation.confirmatory_split
            ].copy()
            if frame.empty:
                raise RuntimeError(
                    f"No confirmatory-split queries for {condition.value}/{model.value}"
                )
            loaded[(condition, model)] = frame
            source_manifests[f"{condition.value}/{model.value}"] = manifest[
                "configuration_fingerprint"
            ]

    configuration = {
        "chunk_size_tokens": chunk_size,
        "confirmatory_split": config.evaluation.confirmatory_split,
        "comparisons": [[first.value, second.value] for first, second in COMPARISONS],
        "models": [model.value for model in selected_models],
        "iterations": config.evaluation.bootstrap_iterations,
        "confidence_level": config.evaluation.confidence_level,
        "metrics": config.evaluation.bootstrap_metrics,
        "seed": config.project.seed,
        "primary_metric": config.evaluation.primary_metric,
        "error_analysis_per_direction": config.evaluation.error_analysis_per_direction,
        "source_manifests": source_manifests,
    }
    fingerprint = _fingerprint(configuration)
    if output_path.exists() and manifest_path.exists() and not overwrite:
        existing = cast(
            dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        if existing.get("configuration_fingerprint") != fingerprint:
            raise RuntimeError(
                f"Cached comparisons at {output_path} do not match current inputs. "
                "Re-run with --overwrite."
            )
        return existing

    output_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in bootstrap_dir.glob("*.npz"):
            path.unlink()

    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    seed_offset = 0
    for model in selected_models:
        for analysis_name, analysis_sets in _analysis_groups(model):
            for first, second in COMPARISONS:
                first_key = (first, None) if first is RetrievalCondition.BM25 else (first, model)
                second_key = (second, model)
                first_frame = loaded[first_key]
                second_frame = loaded[second_key]
                first_subset = first_frame[first_frame["analysis_set"].isin(analysis_sets)]
                second_subset = second_frame[second_frame["analysis_set"].isin(analysis_sets)]
                metadata_columns = [
                    "query_id",
                    "document_id",
                    "question",
                    "query_type",
                    "analysis_set",
                ]
                metric_columns = list(config.evaluation.bootstrap_metrics)
                first_select = first_subset[metadata_columns + metric_columns].copy()
                second_select = second_subset[metadata_columns + metric_columns].copy()
                first_select = first_select.rename(
                    columns={metric: f"first__{metric}" for metric in metric_columns}
                )
                second_select = second_select.rename(
                    columns={metric: f"second__{metric}" for metric in metric_columns}
                )
                merged = first_select.merge(
                    second_select,
                    on=["query_id", "document_id", "question", "query_type", "analysis_set"],
                    how="inner",
                    validate="one_to_one",
                )
                if len(merged) != len(second_subset) or len(merged) != len(first_subset):
                    raise RuntimeError(
                        f"Query sets differ for {first.value} vs {second.value}, "
                        f"{model.value}/{analysis_name}"
                    )

                primary_metric = config.evaluation.primary_metric
                if primary_metric not in metric_columns:
                    primary_metric = metric_columns[0]
                difference_column = f"difference__{primary_metric}"
                merged[difference_column] = (
                    merged[f"first__{primary_metric}"]
                    - merged[f"second__{primary_metric}"]
                )
                count = config.evaluation.error_analysis_per_direction
                extremes = [
                    (
                        "first_better",
                        merged[merged[difference_column] > 0]
                        .sort_values(
                            [difference_column, "query_id"], ascending=[False, True]
                        )
                        .head(count),
                    ),
                    (
                        "second_better",
                        merged[merged[difference_column] < 0]
                        .sort_values(
                            [difference_column, "query_id"], ascending=[True, True]
                        )
                        .head(count),
                    ),
                ]
                for direction, candidates in extremes:
                    for candidate_rank, (_, candidate) in enumerate(
                        candidates.iterrows(), start=1
                    ):
                        candidate_rows.append(
                            {
                                "model_key": model.value,
                                "analysis_set": analysis_name,
                                "first_condition": first.value,
                                "second_condition": second.value,
                                "metric": primary_metric,
                                "direction": direction,
                                "candidate_rank": candidate_rank,
                                "query_id": str(candidate["query_id"]),
                                "document_id": str(candidate["document_id"]),
                                "query_type": str(candidate["query_type"]),
                                "question": str(candidate["question"]),
                                "first_score": float(
                                    candidate[f"first__{primary_metric}"]
                                ),
                                "second_score": float(
                                    candidate[f"second__{primary_metric}"]
                                ),
                                "difference_first_minus_second": float(
                                    candidate[difference_column]
                                ),
                            }
                        )

                for group_type, group_value, group in _group_slices(merged):
                    if group.empty:
                        continue
                    for metric in metric_columns:
                        seed_offset += 1
                        observed, lower, upper, p_value, samples = paired_document_bootstrap(
                            group,
                            first_column=f"first__{metric}",
                            second_column=f"second__{metric}",
                            iterations=config.evaluation.bootstrap_iterations,
                            confidence_level=config.evaluation.confidence_level,
                            seed=config.project.seed + seed_offset,
                        )
                        comparison_id = (
                            f"{model.value}__{analysis_name}__{group_type}__{group_value}__"
                            f"{first.value}__{second.value}__{metric}"
                        )
                        safe_name = hashlib.sha1(
                            comparison_id.encode("utf-8"), usedforsecurity=False
                        ).hexdigest()[:16]
                        bootstrap_file = bootstrap_dir / f"bootstrap_{safe_name}.npz"
                        np.savez_compressed(
                            bootstrap_file,
                            differences=samples,
                            observed_difference=np.asarray([observed], dtype=np.float64),
                        )
                        rows.append(
                            {
                                "model_key": model.value,
                                "analysis_set": analysis_name,
                                "group_type": group_type,
                                "group_value": group_value,
                                "first_condition": first.value,
                                "second_condition": second.value,
                                "metric": metric,
                                "query_count": len(group),
                                "document_count": group["document_id"].nunique(),
                                "first_mean": group[f"first__{metric}"].mean(),
                                "second_mean": group[f"second__{metric}"].mean(),
                                "mean_difference_first_minus_second": observed,
                                "confidence_lower": lower,
                                "confidence_upper": upper,
                                "bootstrap_p_value": p_value,
                                "significant_by_ci": bool(lower > 0.0 or upper < 0.0),
                                "bootstrap_file": str(bootstrap_file.relative_to(output_dir)),
                            }
                        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No pairwise comparisons were generated")
    frame["holm_adjusted_p_value"] = np.nan
    family_columns = ["model_key", "analysis_set", "group_type", "group_value", "metric"]
    for _, indices in frame.groupby(family_columns, dropna=False).groups.items():
        index_list = list(indices)
        adjusted = holm_adjust(frame.loc[index_list, "bootstrap_p_value"].tolist())
        frame.loc[index_list, "holm_adjusted_p_value"] = adjusted
    frame["significant_after_holm_0_05"] = frame["holm_adjusted_p_value"] < 0.05
    frame = frame.sort_values(
        [
            "model_key",
            "analysis_set",
            "group_type",
            "group_value",
            "first_condition",
            "second_condition",
            "metric",
        ]
    )
    frame.to_csv(output_path, index=False)
    candidates_frame = pd.DataFrame(candidate_rows)
    candidates_frame.to_csv(candidates_path, index=False)

    manifest = {
        "schema_version": 1,
        "kind": "paired_document_bootstrap_comparisons",
        "chunk_size_tokens": chunk_size,
        "comparison_count": len(frame),
        "error_analysis_candidate_count": len(candidates_frame),
        "configuration": configuration,
        "configuration_fingerprint": fingerprint,
        "files": {
            "comparisons": output_path.name,
            "error_analysis_candidates": candidates_path.name,
            "bootstrap_directory": bootstrap_dir.name,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


__all__ = ["COMPARISONS", "compare_conditions", "holm_adjust", "paired_document_bootstrap"]
