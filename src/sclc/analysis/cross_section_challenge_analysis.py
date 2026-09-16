from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from sclc.config import AppConfig
from sclc.evaluation.statistics import holm_adjust, paired_document_bootstrap
from sclc.options import EmbeddingModel, RetrievalCondition
from sclc.paths import evaluation_dir

CHALLENGE_CONDITIONS: tuple[RetrievalCondition, ...] = (
    RetrievalCondition.SECTION_ISOLATED,
    RetrievalCondition.SECTION_CONSTRAINED,
    RetrievalCondition.GLOBAL,
)

CHALLENGE_COMPARISONS: tuple[
    tuple[RetrievalCondition, RetrievalCondition], ...
] = (
    (RetrievalCondition.SECTION_ISOLATED, RetrievalCondition.SECTION_CONSTRAINED),
    (RetrievalCondition.SECTION_CONSTRAINED, RetrievalCondition.GLOBAL),
)

_METADATA_COLUMNS = [
    "query_id",
    "document_id",
    "question",
    "query_type",
    "analysis_set",
    "split",
]


def _fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bootstrap_filename(identifier: str) -> str:
    safe_name = hashlib.sha1(
        identifier.encode("utf-8"), usedforsecurity=False
    ).hexdigest()[:16]
    return f"bootstrap_{safe_name}.npz"


def _load_metrics(
    config: AppConfig,
    *,
    condition: RetrievalCondition,
    model: EmbeddingModel,
    chunk_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    directory = evaluation_dir(config, chunk_size) / condition.value / model.value
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Evaluation for {condition.value}/{model.value} at {chunk_size} tokens "
            "is missing. Run challenge encode, retrieve, and evaluate first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics_path = directory / manifest["files"]["query_metrics"]
    frame = pd.read_csv(
        metrics_path,
        dtype={"query_id": "string", "document_id": "string"},
    )
    missing = set(_METADATA_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(
            f"{condition.value}/{model.value}/{chunk_size} query metrics are missing "
            f"columns: {sorted(missing)}"
        )
    return frame, manifest


def _analysis_groups(
    model: EmbeddingModel,
    accepted: pd.DataFrame,
) -> list[tuple[str, set[str]]]:
    cross_model_ids = set(
        accepted.loc[
            accepted["eligibility_group"].eq("cross_model_core"), "query_id"
        ].astype(str)
    )
    if model is EmbeddingModel.JINA:
        return [("accepted_cross_model", cross_model_ids)]
    all_ids = set(accepted["query_id"].astype(str))
    return [
        ("accepted_all", all_ids),
        ("accepted_cross_model", cross_model_ids),
    ]


def _merge_pair(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    first_condition: RetrievalCondition,
    second_condition: RetrievalCondition,
    metrics: Sequence[str],
) -> pd.DataFrame:
    first_select = first[_METADATA_COLUMNS + list(metrics)].copy().rename(
        columns={metric: f"first__{metric}" for metric in metrics}
    )
    second_select = second[_METADATA_COLUMNS + list(metrics)].copy().rename(
        columns={metric: f"second__{metric}" for metric in metrics}
    )
    merged = first_select.merge(
        second_select,
        on=_METADATA_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(first) or len(merged) != len(second):
        raise RuntimeError(
            "Accepted challenge query sets differ for "
            f"{first_condition.value} and {second_condition.value}"
        )
    return merged


def analyse_cross_section_challenge(
    config: AppConfig,
    *,
    accepted_queries_path: Path,
    model: EmbeddingModel,
    chunk_sizes: Sequence[int],
    overwrite: bool = False,
) -> dict[str, Any]:
    sizes = tuple(dict.fromkeys(int(size) for size in chunk_sizes))
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("Retrieval-unit sizes must be positive integers")
    unsupported = set(sizes).difference(config.chunking.supported_chunk_sizes)
    if unsupported:
        raise ValueError(
            f"Unsupported retrieval-unit sizes {sorted(unsupported)}; expected a subset of "
            f"{config.chunking.supported_chunk_sizes}"
        )
    if not accepted_queries_path.exists():
        raise FileNotFoundError(accepted_queries_path)

    accepted = pd.read_csv(
        accepted_queries_path,
        dtype={"query_id": "string", "document_id": "string"},
    )
    required = {"query_id", "document_id", "eligibility_group", "include"}
    missing = required.difference(accepted.columns)
    if missing:
        raise ValueError(
            f"Accepted challenge file is missing columns: {sorted(missing)}"
        )
    accepted = accepted[accepted["include"].astype(str).str.lower().eq("yes")].copy()
    if accepted.empty:
        raise RuntimeError("The accepted challenge set is empty")
    if accepted["query_id"].duplicated().any():
        raise ValueError("Accepted challenge query IDs must be unique")

    metrics = tuple(config.evaluation.bootstrap_metrics)
    output_dir = config.paths.analysis_dir / "cross_section_challenge_results" / model.value
    summary_path = output_dir / "summary_by_retrieval_unit_size.csv"
    comparisons_path = output_dir / "comparisons_within_retrieval_unit_size.csv"
    effects_path = output_dir / "query_scope_effects.csv"
    manifest_path = output_dir / "manifest.json"
    bootstrap_dir = output_dir / "bootstrap"

    loaded: dict[tuple[int, RetrievalCondition], pd.DataFrame] = {}
    source_manifests: dict[str, str] = {}
    for size in sizes:
        for condition in CHALLENGE_CONDITIONS:
            frame, manifest = _load_metrics(
                config,
                condition=condition,
                model=model,
                chunk_size=size,
            )
            missing_metrics = set(metrics).difference(frame.columns)
            if missing_metrics:
                raise ValueError(
                    f"{condition.value}/{model.value}/{size} is missing metrics: "
                    f"{sorted(missing_metrics)}"
                )
            loaded[(size, condition)] = frame
            source_manifests[f"{size}/{condition.value}"] = manifest[
                "configuration_fingerprint"
            ]

    configuration = {
        "model": model.value,
        "chunk_sizes": list(sizes),
        "conditions": [condition.value for condition in CHALLENGE_CONDITIONS],
        "comparisons": [
            [first.value, second.value]
            for first, second in CHALLENGE_COMPARISONS
        ],
        "analysis_groups": [
            name for name, _ in _analysis_groups(model, accepted)
        ],
        "metrics": list(metrics),
        "iterations": config.evaluation.bootstrap_iterations,
        "confidence_level": config.evaluation.confidence_level,
        "seed": config.project.seed,
        "accepted_queries_sha256": _file_sha256(accepted_queries_path),
        "source_manifests": source_manifests,
    }
    fingerprint = _fingerprint(configuration)
    if manifest_path.exists() and not overwrite:
        existing = cast(
            dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        if existing.get("configuration_fingerprint") != fingerprint:
            raise RuntimeError(
                f"Cached challenge analysis at {output_dir} does not match the "
                "current inputs. Re-run with overwrite=True."
            )
        return existing

    output_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in bootstrap_dir.glob("*.npz"):
            path.unlink()

    summary_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    seed_offset = 0

    for group_name, group_ids in _analysis_groups(model, accepted):
        if not group_ids:
            continue
        for size in sizes:
            filtered: dict[RetrievalCondition, pd.DataFrame] = {}
            for condition in CHALLENGE_CONDITIONS:
                frame = loaded[(size, condition)]
                sample = frame[frame["query_id"].astype(str).isin(group_ids)].copy()
                observed_ids = set(sample["query_id"].astype(str))
                missing_ids = group_ids.difference(observed_ids)
                if missing_ids:
                    raise RuntimeError(
                        f"{condition.value}/{model.value}/{size} is missing "
                        f"{len(missing_ids)} accepted challenge queries; first: "
                        f"{sorted(missing_ids)[:5]}"
                    )
                if len(sample) != len(group_ids):
                    raise RuntimeError(
                        f"Duplicate challenge metrics for {condition.value}/"
                        f"{model.value}/{size}"
                    )
                filtered[condition] = sample
                row: dict[str, Any] = {
                    "model_key": model.value,
                    "analysis_group": group_name,
                    "chunk_size_tokens": size,
                    "condition": condition.value,
                    "query_count": len(sample),
                    "document_count": sample["document_id"].nunique(),
                }
                for metric in metrics:
                    row[metric] = float(sample[metric].mean())
                summary_rows.append(row)

            for first, second in CHALLENGE_COMPARISONS:
                merged = _merge_pair(
                    filtered[first],
                    filtered[second],
                    first_condition=first,
                    second_condition=second,
                    metrics=metrics,
                )
                effect_frame = merged[_METADATA_COLUMNS].copy()
                effect_frame["model_key"] = model.value
                effect_frame["analysis_group"] = group_name
                effect_frame["chunk_size_tokens"] = size
                effect_frame["effect_name"] = (
                    f"{first.value}_minus_{second.value}"
                )
                for metric in metrics:
                    effect_frame[metric] = (
                        merged[f"first__{metric}"] - merged[f"second__{metric}"]
                    )
                effect_rows.extend(effect_frame.to_dict(orient="records"))

                for metric in metrics:
                    seed_offset += 1
                    observed, lower, upper, p_value, samples = paired_document_bootstrap(
                        merged,
                        first_column=f"first__{metric}",
                        second_column=f"second__{metric}",
                        iterations=config.evaluation.bootstrap_iterations,
                        confidence_level=config.evaluation.confidence_level,
                        seed=config.project.seed + seed_offset,
                    )
                    identifier = (
                        f"challenge__{model.value}__{group_name}__{size}__"
                        f"{first.value}__{second.value}__{metric}"
                    )
                    bootstrap_file = bootstrap_dir / _bootstrap_filename(identifier)
                    np.savez_compressed(
                        bootstrap_file,
                        differences=samples,
                        observed_difference=np.asarray([observed], dtype=np.float64),
                    )
                    comparison_rows.append(
                        {
                            "model_key": model.value,
                            "analysis_group": group_name,
                            "chunk_size_tokens": size,
                            "first_condition": first.value,
                            "second_condition": second.value,
                            "metric": metric,
                            "query_count": len(merged),
                            "document_count": merged["document_id"].nunique(),
                            "first_mean": float(merged[f"first__{metric}"].mean()),
                            "second_mean": float(merged[f"second__{metric}"].mean()),
                            "mean_difference_first_minus_second": observed,
                            "confidence_lower": lower,
                            "confidence_upper": upper,
                            "bootstrap_p_value": p_value,
                            "significant_by_ci": bool(lower > 0.0 or upper < 0.0),
                            "bootstrap_file": str(
                                bootstrap_file.relative_to(output_dir)
                            ),
                        }
                    )

    summary = pd.DataFrame(summary_rows)
    comparisons = pd.DataFrame(comparison_rows)
    effects = pd.DataFrame(effect_rows)
    if summary.empty or comparisons.empty or effects.empty:
        raise RuntimeError("Cross-section challenge analysis generated empty outputs")

    comparisons["holm_adjusted_p_value"] = np.nan
    family = [
        "model_key",
        "analysis_group",
        "chunk_size_tokens",
        "metric",
    ]
    for _, indices in comparisons.groupby(family, dropna=False).groups.items():
        index_list = list(indices)
        comparisons.loc[index_list, "holm_adjusted_p_value"] = holm_adjust(
            comparisons.loc[index_list, "bootstrap_p_value"].tolist()
        )
    comparisons["significant_after_holm_0_05"] = (
        comparisons["holm_adjusted_p_value"] < 0.05
    )

    summary.sort_values(
        ["model_key", "analysis_group", "chunk_size_tokens", "condition"]
    ).to_csv(summary_path, index=False)
    comparisons.sort_values(
        [
            "model_key",
            "analysis_group",
            "chunk_size_tokens",
            "metric",
            "first_condition",
            "second_condition",
        ]
    ).to_csv(comparisons_path, index=False)
    effects.sort_values(
        [
            "model_key",
            "analysis_group",
            "effect_name",
            "chunk_size_tokens",
            "document_id",
            "query_id",
        ]
    ).to_csv(effects_path, index=False)

    manifest = {
        "schema_version": 1,
        "kind": "cross_section_challenge_analysis",
        "model_key": model.value,
        "chunk_sizes": list(sizes),
        "configuration": configuration,
        "configuration_fingerprint": fingerprint,
        "summary_row_count": len(summary),
        "comparison_count": len(comparisons),
        "scope_effect_row_count": len(effects),
        "files": {
            "summary_by_retrieval_unit_size": summary_path.name,
            "comparisons_within_retrieval_unit_size": comparisons_path.name,
            "query_scope_effects": effects_path.name,
            "bootstrap_directory": bootstrap_dir.name,
        },
        "interpretation_notes": [
            "The challenge set is a secondary exploratory analysis.",
            "Granite accepted_all contains all accepted questions; the common "
            "accepted_cross_model group is used for direct Granite/Jina comparison.",
            "Within-size differences are first condition minus second condition.",
            "Holm correction is applied across the two predefined scope comparisons "
            "within each model, analysis group, retrieval-unit size, and metric.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


__all__ = [
    "CHALLENGE_COMPARISONS",
    "CHALLENGE_CONDITIONS",
    "analyse_cross_section_challenge",
]
