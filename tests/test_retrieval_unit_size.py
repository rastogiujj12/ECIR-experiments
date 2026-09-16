import json
from pathlib import Path

import pandas as pd
import pytest

from sclc.analysis.retrieval_unit_size import (
    SCOPE_EFFECTS,
    analyse_retrieval_unit_size,
)
from sclc.config import AppConfig
from sclc.options import EmbeddingModel, RetrievalCondition
from sclc.paths import evaluation_dir


def test_scope_effect_directions_match_dissertation() -> None:
    assert SCOPE_EFFECTS == (
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


def make_config(tmp_path: Path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "project": {"seed": 11},
            "paths": {
                "processed_dir": tmp_path / "processed",
                "profile_dir": tmp_path / "profiles",
                "subset_dir": tmp_path / "subsets",
                "retrieval_unit_dir": tmp_path / "retrieval_units",
                "encoding_dir": tmp_path / "encodings",
                "ranking_dir": tmp_path / "rankings",
                "evaluation_dir": tmp_path / "evaluation",
                "analysis_dir": tmp_path / "analysis",
                "hf_cache_dir": tmp_path / "cache",
            },
            "dataset": {"repo_id": "allenai/qasper"},
            "document": {},
            "chunking": {
                "canonical_tokenizer": "granite",
                "chunk_size_tokens": 128,
                "supported_chunk_sizes": [128, 256, 512],
            },
            "models": {
                "granite": {"model_id": "granite", "max_document_tokens": 32768},
                "jina": {"model_id": "jina", "max_document_tokens": 8192},
            },
            "sampling": {},
            "evaluation": {
                "bootstrap_iterations": 50,
                "bootstrap_metrics": [
                    "ndcg_at_5",
                    "evidence_paragraph_recall_at_token_budget_1024",
                ],
                "primary_metric": "ndcg_at_5",
                "confirmatory_split": "test",
            },
        }
    )


def write_evaluation(
    config: AppConfig,
    *,
    size: int,
    condition: str,
    scores: dict[str, tuple[float, float]],
) -> None:
    directory = evaluation_dir(config, size) / condition / "granite"
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    metadata = {
        "q1": ("d1", "train"),
        "q2": ("d1", "test"),
        "q3": ("d2", "test"),
    }
    for query_id, (document_id, split) in metadata.items():
        ndcg, budget = scores[query_id]
        rows.append(
            {
                "query_id": query_id,
                "document_id": document_id,
                "question": query_id,
                "query_type": "factual",
                "analysis_set": "cross_model_core",
                "split": split,
                "ndcg_at_5": ndcg,
                "evidence_paragraph_recall_at_token_budget_1024": budget,
            }
        )
    pd.DataFrame(rows).to_csv(directory / "query_metrics.csv", index=False)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "configuration_fingerprint": f"{size}-{condition}",
                "files": {"query_metrics": "query_metrics.csv"},
            }
        ),
        encoding="utf-8",
    )


def test_retrieval_unit_size_uses_all_and_test_queries(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    base_by_condition = {
        "fixed_dense": 0.7,
        "section_isolated": 0.6,
        "section_constrained": 0.5,
        "global": 0.3,
    }
    for size in (128, 256, 512):
        size_shift = {128: 0.0, 256: 0.02, 512: 0.04}[size]
        for condition, base in base_by_condition.items():
            # Make the section-constrained minus global effect decrease as chunks grow.
            condition_shift = 0.0
            if condition == "global":
                condition_shift = {128: 0.0, 256: 0.05, 512: 0.10}[size]
            scores = {
                query_id: (
                    base + size_shift + condition_shift + offset,
                    base + size_shift + condition_shift + offset,
                )
                for query_id, offset in {"q1": 0.0, "q2": 0.01, "q3": -0.01}.items()
            }
            write_evaluation(
                config,
                size=size,
                condition=condition,
                scores=scores,
            )

    manifest = analyse_retrieval_unit_size(
        config,
        model=EmbeddingModel.GRANITE,
        chunk_sizes=(128, 256, 512),
    )
    output_dir = config.paths.analysis_dir / "retrieval_unit_size" / "granite"
    summary = pd.read_csv(output_dir / manifest["files"]["summary_by_retrieval_unit_size"])
    comparisons = pd.read_csv(
        output_dir / manifest["files"]["comparisons_within_retrieval_unit_size"]
    )
    interactions = pd.read_csv(
        output_dir / manifest["files"]["scope_interactions_across_retrieval_unit_sizes"]
    )

    assert set(summary["sample_scope"]) == {"all_questions", "split_test"}
    all_questions = summary[
        (summary["analysis_set"] == "cross_model_core")
        & (summary["sample_scope"] == "all_questions")
    ]
    assert set(all_questions["query_count"]) == {3}
    test_questions = summary[
        (summary["analysis_set"] == "cross_model_core")
        & (summary["sample_scope"] == "split_test")
    ]
    assert set(test_questions["query_count"]) == {2}
    assert set(comparisons["chunk_size_tokens"]) == {128, 256, 512}
    assert "mean_difference_first_minus_second" in comparisons.columns

    central = interactions[
        (interactions["analysis_set"] == "cross_model_core")
        & (interactions["sample_scope"] == "all_questions")
        & (interactions["effect_name"] == "section_constrained_minus_global")
        & (interactions["metric"] == "ndcg_at_5")
        & (interactions["first_chunk_size_tokens"] == 128)
        & (interactions["second_chunk_size_tokens"] == 512)
    ]
    assert len(central) == 1
    assert central.iloc[0][
        "change_in_effect_second_size_minus_first_size"
    ] == pytest.approx(-0.1)
    assert list((output_dir / "bootstrap").glob("*.npz"))
