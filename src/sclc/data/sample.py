from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sclc.config import AppConfig


def read_profile_csv(path: Path) -> pd.DataFrame:
    """Read the profiling manifest without coercing document IDs to numbers.

    QASPER document IDs resemble decimal values (for example, 1610.06510).
    Pandas would otherwise infer them as floats and remove trailing zeros.
    """
    return pd.read_csv(path, dtype={"document_id": "string"})


def _add_length_strata(frame: pd.DataFrame, bins: int) -> pd.DataFrame:
    result = frame.copy()
    unique_lengths = result["granite_tokens"].nunique()
    effective_bins = min(bins, unique_lengths)
    if effective_bins < 2:
        result["length_stratum"] = "all"
        return result

    categories = pd.qcut(
        result["granite_tokens"],
        q=effective_bins,
        duplicates="drop",
    )
    codes = categories.cat.codes
    result["length_stratum"] = codes.map(lambda value: f"length_{value + 1}")
    return result


def _stratified_sample(
    frame: pd.DataFrame,
    target_size: int,
    seed: int,
    bins: int,
) -> pd.DataFrame:
    if target_size <= 0 or frame.empty:
        return frame.iloc[0:0].copy()
    if len(frame) <= target_size:
        return _add_length_strata(frame, bins)

    stratified = _add_length_strata(frame, bins)
    rng = np.random.default_rng(seed)

    groups = list(stratified.groupby(["split", "length_stratum"], dropna=False))
    proportions = {key: len(group) / len(stratified) for key, group in groups}

    allocations = {
        key: min(len(group), int(np.floor(target_size * proportions[key])))
        for key, group in groups
    }

    remaining = target_size - sum(allocations.values())
    ranked_keys = sorted(
        groups,
        key=lambda item: (
            target_size * proportions[item[0]] - allocations[item[0]],
            len(item[1]),
        ),
        reverse=True,
    )

    for key, group in ranked_keys:
        if remaining <= 0:
            break
        capacity = len(group) - allocations[key]
        if capacity > 0:
            allocations[key] += 1
            remaining -= 1

    sampled_parts = []
    for key, group in groups:
        take = allocations[key]
        if take <= 0:
            continue
        indices = rng.choice(group.index.to_numpy(), size=take, replace=False)
        sampled_parts.append(stratified.loc[indices])

    sampled = pd.concat(sampled_parts, ignore_index=True)
    return sampled.sort_values(["eligibility_group", "length_stratum", "document_id"])


def select_documents(config: AppConfig, profile: pd.DataFrame) -> pd.DataFrame:
    eligible = profile[
        profile["usable_question_count"] >= config.sampling.minimum_usable_questions
    ].copy()

    core = eligible[eligible["eligibility_group"] == "cross_model_core"]
    extended = eligible[eligible["eligibility_group"] == "granite_extended"]

    selected_core = _stratified_sample(
        core,
        target_size=config.sampling.core_documents,
        seed=config.project.seed,
        bins=config.sampling.length_bins,
    )
    selected_core["analysis_set"] = "cross_model_core"

    selected_extended = _stratified_sample(
        extended,
        target_size=config.sampling.granite_extended_documents,
        seed=config.project.seed + 1,
        bins=config.sampling.length_bins,
    )
    selected_extended["analysis_set"] = "granite_extended"

    selected = pd.concat([selected_core, selected_extended], ignore_index=True)
    return selected.sort_values(["analysis_set", "length_stratum", "document_id"])
