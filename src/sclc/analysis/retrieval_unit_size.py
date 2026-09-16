from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

from sclc.config import AppConfig
from sclc.evaluation.statistics import holm_adjust, paired_document_bootstrap
from sclc.options import EmbeddingModel, RetrievalCondition
from sclc.paths import evaluation_dir

DENSE_CONDITIONS: tuple[RetrievalCondition, ...] = (
    RetrievalCondition.FIXED_DENSE,
    RetrievalCondition.SECTION_ISOLATED,
    RetrievalCondition.SECTION_CONSTRAINED,
    RetrievalCondition.GLOBAL,
)

WITHIN_SIZE_COMPARISONS: tuple[
    tuple[RetrievalCondition, RetrievalCondition], ...
] = (
    # Match the signed directions used in the dissertation's primary table.
    (RetrievalCondition.SECTION_ISOLATED, RetrievalCondition.FIXED_DENSE),
    (RetrievalCondition.SECTION_CONSTRAINED, RetrievalCondition.SECTION_ISOLATED),
    (RetrievalCondition.GLOBAL, RetrievalCondition.SECTION_CONSTRAINED),
)

SCOPE_EFFECTS: tuple[
    tuple[str, RetrievalCondition, RetrievalCondition], ...
] = (
    (
        "section_isolated_minus_section_constrained",
        RetrievalCondition.SECTION_ISOLATED,
        RetrievalCondition.SECTION_CONSTRAINED,
    ),
    (
        "section_constrained_minus_global",
        RetrievalCondition.SECTION_CONSTRAINED,
        RetrievalCondition.GLOBAL,
    ),
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


def _sample_slices(
    frame: pd.DataFrame,
    *,
    confirmatory_split: str,
) -> list[tuple[str, pd.DataFrame]]:
    slices = [("all_questions", frame)]
    test = frame[frame["split"] == confirmatory_split].copy()
    if not test.empty:
        slices.append((f"split_{confirmatory_split}", test))
    return slices


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
            "is missing. Run encode, retrieve, and evaluate first."
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


def _merge_condition_pair(
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
            "Query sets differ for "
            f"{first_condition.value} and {second_condition.value}"
        )
    return merged


def _bootstrap_filename(identifier: str) -> str:
    safe_name = hashlib.sha1(
        identifier.encode("utf-8"), usedforsecurity=False
    ).hexdigest()[:16]
    return f"bootstrap_{safe_name}.npz"


def analyse_retrieval_unit_size(
    config: AppConfig,
    *,
    model: EmbeddingModel,
    chunk_sizes: Sequence[int],
    overwrite: bool = False,
) -> dict[str, Any]:
    sizes = tuple(dict.fromkeys(int(size) for size in chunk_sizes))
    if len(sizes) < 2:
        raise ValueError("Retrieval-unit-size analysis requires at least two retrieval-unit sizes")
    if any(size <= 0 for size in sizes):
        raise ValueError("Retrieval-unit sizes must be positive integers")
    unsupported = set(sizes).difference(config.chunking.supported_chunk_sizes)
    if unsupported:
        raise ValueError(
            f"Unsupported retrieval-unit sizes {sorted(unsupported)}; expected a subset of "
            f"{config.chunking.supported_chunk_sizes}"
        )

    metrics = tuple(config.evaluation.bootstrap_metrics)
    output_dir = config.paths.analysis_dir / "retrieval_unit_size" / model.value
    summary_path = output_dir / "summary_by_retrieval_unit_size.csv"
    comparisons_path = output_dir / "comparisons_within_retrieval_unit_size.csv"
    effects_path = output_dir / "query_scope_effects.csv"
    interactions_path = output_dir / "scope_interactions_across_retrieval_unit_sizes.csv"
    manifest_path = output_dir / "manifest.json"
    bootstrap_dir = output_dir / "bootstrap"

    loaded: dict[tuple[int, RetrievalCondition], pd.DataFrame] = {}
    source_manifests: dict[str, str] = {}
    for size in sizes:
        expected_query_ids: set[str] | None = None
        for condition in DENSE_CONDITIONS:
            frame, manifest = _load_metrics(
                config,
                condition=condition,
                model=model,
                chunk_size=size,
            )
            missing_metrics = set(metrics).difference(frame.columns)
            if missing_metrics:
                raise ValueError(
                    f"{condition.value}/{model.value}/{size} is missing configured "
                    f"bootstrap metrics: {sorted(missing_metrics)}"
                )
            query_ids = set(frame["query_id"].astype(str))
            if expected_query_ids is None:
                expected_query_ids = query_ids
            elif query_ids != expected_query_ids:
                raise RuntimeError(
                    f"Dense condition query sets differ at retrieval-unit size {size}"
                )
            loaded[(size, condition)] = frame
            source_manifests[f"{size}/{condition.value}"] = manifest[
                "configuration_fingerprint"
            ]

    reference_ids = set(loaded[(sizes[0], DENSE_CONDITIONS[0])]["query_id"].astype(str))
    for size in sizes[1:]:
        size_ids = set(loaded[(size, DENSE_CONDITIONS[0])]["query_id"].astype(str))
        if size_ids != reference_ids:
            raise RuntimeError(
                "Evaluated query sets differ across retrieval-unit sizes; "
                "retrieval-unit-size analysis requires identical questions at every size"
            )

    configuration = {
        "model": model.value,
        "chunk_sizes": list(sizes),
        "conditions": [condition.value for condition in DENSE_CONDITIONS],
        "within_size_comparisons": [
            [first.value, second.value]
            for first, second in WITHIN_SIZE_COMPARISONS
        ],
        "scope_effects": [
            {
                "name": name,
                "first_condition": first.value,
                "second_condition": second.value,
                "effect": "first_minus_second",
            }
            for name, first, second in SCOPE_EFFECTS
        ],
        "sample_scopes": [
            "all_questions",
            f"split_{config.evaluation.confirmatory_split}",
        ],
        "analysis_groups": [name for name, _ in _analysis_groups(model)],
        "metrics": list(metrics),
        "iterations": config.evaluation.bootstrap_iterations,
        "confidence_level": config.evaluation.confidence_level,
        "seed": config.project.seed,
        "source_manifests": source_manifests,
    }
    fingerprint = _fingerprint(configuration)
    if manifest_path.exists() and not overwrite:
        existing = cast(
            dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        if existing.get("configuration_fingerprint") != fingerprint:
            raise RuntimeError(
                f"Cached retrieval-unit-size analysis at {output_dir} does not match current "
                "inputs. Re-run with --overwrite."
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
    interaction_rows: list[dict[str, Any]] = []
    seed_offset = 0

    for analysis_name, analysis_sets in _analysis_groups(model):
        for size in sizes:
            for condition in DENSE_CONDITIONS:
                frame = loaded[(size, condition)]
                analysis_frame = frame[frame["analysis_set"].isin(analysis_sets)].copy()
                for sample_scope, sample in _sample_slices(
                    analysis_frame,
                    confirmatory_split=config.evaluation.confirmatory_split,
                ):
                    if sample.empty:
                        continue
                    row: dict[str, Any] = {
                        "model_key": model.value,
                        "analysis_set": analysis_name,
                        "sample_scope": sample_scope,
                        "chunk_size_tokens": size,
                        "condition": condition.value,
                        "query_count": len(sample),
                        "document_count": sample["document_id"].nunique(),
                    }
                    for metric in metrics:
                        row[metric] = float(sample[metric].mean())
                    summary_rows.append(row)

            for first, second in WITHIN_SIZE_COMPARISONS:
                first_frame = loaded[(size, first)]
                second_frame = loaded[(size, second)]
                first_analysis = first_frame[
                    first_frame["analysis_set"].isin(analysis_sets)
                ].copy()
                second_analysis = second_frame[
                    second_frame["analysis_set"].isin(analysis_sets)
                ].copy()
                merged = _merge_condition_pair(
                    first_analysis,
                    second_analysis,
                    first_condition=first,
                    second_condition=second,
                    metrics=metrics,
                )
                for sample_scope, sample in _sample_slices(
                    merged,
                    confirmatory_split=config.evaluation.confirmatory_split,
                ):
                    if sample.empty:
                        continue
                    for metric in metrics:
                        seed_offset += 1
                        observed, lower, upper, p_value, samples = paired_document_bootstrap(
                            sample,
                            first_column=f"first__{metric}",
                            second_column=f"second__{metric}",
                            iterations=config.evaluation.bootstrap_iterations,
                            confidence_level=config.evaluation.confidence_level,
                            seed=config.project.seed + seed_offset,
                        )
                        identifier = (
                            f"within__{model.value}__{analysis_name}__{sample_scope}__"
                            f"{size}__{first.value}__{second.value}__{metric}"
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
                                "analysis_set": analysis_name,
                                "sample_scope": sample_scope,
                                "chunk_size_tokens": size,
                                "first_condition": first.value,
                                "second_condition": second.value,
                                "metric": metric,
                                "query_count": len(sample),
                                "document_count": sample["document_id"].nunique(),
                                "first_mean": float(sample[f"first__{metric}"].mean()),
                                "second_mean": float(sample[f"second__{metric}"].mean()),
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

        # Build per-query scope effects at each size, then test whether those effects
        # change across retrieval-unit sizes (the size-by-scope interaction).
        effects_by_key: dict[tuple[str, str], pd.DataFrame] = {}
        for effect_name, first, second in SCOPE_EFFECTS:
            for size in sizes:
                first_frame = loaded[(size, first)]
                second_frame = loaded[(size, second)]
                first_analysis = first_frame[
                    first_frame["analysis_set"].isin(analysis_sets)
                ].copy()
                second_analysis = second_frame[
                    second_frame["analysis_set"].isin(analysis_sets)
                ].copy()
                merged = _merge_condition_pair(
                    first_analysis,
                    second_analysis,
                    first_condition=first,
                    second_condition=second,
                    metrics=metrics,
                )
                effect_frame = merged[_METADATA_COLUMNS].copy()
                effect_frame["model_key"] = model.value
                effect_frame["analysis_group"] = analysis_name
                effect_frame["chunk_size_tokens"] = size
                effect_frame["effect_name"] = effect_name
                for metric in metrics:
                    effect_frame[metric] = (
                        merged[f"first__{metric}"] - merged[f"second__{metric}"]
                    )
                effect_rows.extend(effect_frame.to_dict(orient="records"))
                effects_by_key[(effect_name, str(size))] = effect_frame

            for sample_scope in (
                "all_questions",
                f"split_{config.evaluation.confirmatory_split}",
            ):
                for first_size_index, first_size in enumerate(sizes[:-1]):
                    for second_size in sizes[first_size_index + 1 :]:
                        first_effect = effects_by_key[(effect_name, str(first_size))]
                        second_effect = effects_by_key[(effect_name, str(second_size))]
                        if sample_scope != "all_questions":
                            split = sample_scope.removeprefix("split_")
                            first_effect = first_effect[
                                first_effect["split"] == split
                            ].copy()
                            second_effect = second_effect[
                                second_effect["split"] == split
                            ].copy()
                        first_select = first_effect[_METADATA_COLUMNS + list(metrics)].rename(
                            columns={metric: f"first__{metric}" for metric in metrics}
                        )
                        second_select = second_effect[
                            _METADATA_COLUMNS + list(metrics)
                        ].rename(
                            columns={metric: f"second__{metric}" for metric in metrics}
                        )
                        merged_effects = first_select.merge(
                            second_select,
                            on=_METADATA_COLUMNS,
                            how="inner",
                            validate="one_to_one",
                        )
                        if (
                            len(merged_effects) != len(first_effect)
                            or len(merged_effects) != len(second_effect)
                        ):
                            raise RuntimeError(
                                f"Scope-effect query sets differ between {first_size} "
                                f"and {second_size} tokens"
                            )
                        if merged_effects.empty:
                            continue
                        for metric in metrics:
                            seed_offset += 1
                            observed, lower, upper, p_value, samples = (
                                paired_document_bootstrap(
                                    merged_effects,
                                    first_column=f"second__{metric}",
                                    second_column=f"first__{metric}",
                                    iterations=config.evaluation.bootstrap_iterations,
                                    confidence_level=config.evaluation.confidence_level,
                                    seed=config.project.seed + seed_offset,
                                )
                            )
                            identifier = (
                                f"interaction__{model.value}__{analysis_name}__"
                                f"{sample_scope}__{effect_name}__{first_size}__"
                                f"{second_size}__{metric}"
                            )
                            bootstrap_file = bootstrap_dir / _bootstrap_filename(identifier)
                            np.savez_compressed(
                                bootstrap_file,
                                differences=samples,
                                observed_difference=np.asarray(
                                    [observed], dtype=np.float64
                                ),
                            )
                            interaction_rows.append(
                                {
                                    "model_key": model.value,
                                    "analysis_set": analysis_name,
                                    "sample_scope": sample_scope,
                                    "effect_name": effect_name,
                                    "first_chunk_size_tokens": first_size,
                                    "second_chunk_size_tokens": second_size,
                                    "metric": metric,
                                    "query_count": len(merged_effects),
                                    "document_count": merged_effects[
                                        "document_id"
                                    ].nunique(),
                                    "first_effect_mean": float(
                                        merged_effects[f"first__{metric}"].mean()
                                    ),
                                    "second_effect_mean": float(
                                        merged_effects[f"second__{metric}"].mean()
                                    ),
                                    "change_in_effect_second_size_minus_first_size": observed,
                                    "confidence_lower": lower,
                                    "confidence_upper": upper,
                                    "bootstrap_p_value": p_value,
                                    "significant_by_ci": bool(
                                        lower > 0.0 or upper < 0.0
                                    ),
                                    "bootstrap_file": str(
                                        bootstrap_file.relative_to(output_dir)
                                    ),
                                }
                            )

    summary = pd.DataFrame(summary_rows)
    comparisons = pd.DataFrame(comparison_rows)
    effects = pd.DataFrame(effect_rows)
    interactions = pd.DataFrame(interaction_rows)
    if summary.empty or comparisons.empty or effects.empty or interactions.empty:
        raise RuntimeError("Retrieval-unit-size analysis generated empty outputs")

    comparisons["holm_adjusted_p_value"] = np.nan
    within_family = [
        "model_key",
        "analysis_set",
        "sample_scope",
        "chunk_size_tokens",
        "metric",
    ]
    for _, indices in comparisons.groupby(within_family, dropna=False).groups.items():
        index_list = list(indices)
        comparisons.loc[index_list, "holm_adjusted_p_value"] = holm_adjust(
            comparisons.loc[index_list, "bootstrap_p_value"].tolist()
        )
    comparisons["significant_after_holm_0_05"] = (
        comparisons["holm_adjusted_p_value"] < 0.05
    )

    interactions["holm_adjusted_p_value"] = np.nan
    interaction_family = [
        "model_key",
        "analysis_set",
        "sample_scope",
        "metric",
    ]
    for _, indices in interactions.groupby(
        interaction_family, dropna=False
    ).groups.items():
        index_list = list(indices)
        interactions.loc[index_list, "holm_adjusted_p_value"] = holm_adjust(
            interactions.loc[index_list, "bootstrap_p_value"].tolist()
        )
    interactions["significant_after_holm_0_05"] = (
        interactions["holm_adjusted_p_value"] < 0.05
    )

    summary.sort_values(
        [
            "model_key",
            "analysis_set",
            "sample_scope",
            "chunk_size_tokens",
            "condition",
        ]
    ).to_csv(summary_path, index=False)
    comparisons.sort_values(
        [
            "model_key",
            "analysis_set",
            "sample_scope",
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
    interactions.sort_values(
        [
            "model_key",
            "analysis_set",
            "sample_scope",
            "metric",
            "effect_name",
            "first_chunk_size_tokens",
            "second_chunk_size_tokens",
        ]
    ).to_csv(interactions_path, index=False)

    manifest = {
        "schema_version": 1,
        "kind": "retrieval_unit_size_analysis",
        "model_key": model.value,
        "chunk_sizes": list(sizes),
        "query_count": len(reference_ids),
        "summary_row_count": len(summary),
        "comparison_count": len(comparisons),
        "scope_effect_row_count": len(effects),
        "interaction_count": len(interactions),
        "configuration": configuration,
        "configuration_fingerprint": fingerprint,
        "files": {
            "summary_by_retrieval_unit_size": summary_path.name,
            "comparisons_within_retrieval_unit_size": comparisons_path.name,
            "query_scope_effects": effects_path.name,
            "scope_interactions_across_retrieval_unit_sizes": interactions_path.name,
            "bootstrap_directory": bootstrap_dir.name,
        },
        "interpretation_notes": [
            "The all_questions scope includes every eligible question associated with "
            "the frozen selected-paper corpus.",
            "The split_test scope preserves the original held-out test subset.",
            "Within-size differences are first condition minus second condition.",
            "Retrieval-unit-size interactions test whether a scope difference changes between "
            "two retrieval-unit sizes; positive values mean the named effect is larger at the "
            "second retrieval-unit size.",
            "Fixed-rank metrics change the amount of retrieved text as retrieval-unit size "
            "changes; token-budget metrics provide the fairer cross-size comparison.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


__all__ = [
    "DENSE_CONDITIONS",
    "SCOPE_EFFECTS",
    "WITHIN_SIZE_COMPARISONS",
    "analyse_retrieval_unit_size",
]
