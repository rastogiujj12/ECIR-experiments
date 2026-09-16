import pandas as pd
import pytest

from sclc.evaluation.statistics import (
    COMPARISONS,
    holm_adjust,
    paired_document_bootstrap,
)
from sclc.options import RetrievalCondition


def test_primary_comparison_directions_match_dissertation() -> None:
    assert COMPARISONS == (
        (RetrievalCondition.BM25, RetrievalCondition.FIXED_DENSE),
        (RetrievalCondition.SECTION_ISOLATED, RetrievalCondition.FIXED_DENSE),
        (RetrievalCondition.SECTION_CONSTRAINED, RetrievalCondition.SECTION_ISOLATED),
        (RetrievalCondition.SECTION_CONSTRAINED, RetrievalCondition.LENGTH_MATCHED),
        (RetrievalCondition.GLOBAL, RetrievalCondition.SECTION_CONSTRAINED),
    )


def test_holm_adjustment_is_monotonic_in_sorted_order() -> None:
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_document_bootstrap_uses_first_minus_second_difference() -> None:
    frame = pd.DataFrame(
        {
            "document_id": ["d1", "d1", "d2"],
            "first": [0.0, 1.0, 2.0],
            "second": [1.0, 2.0, 3.0],
        }
    )
    observed, lower, upper, p_value, samples = paired_document_bootstrap(
        frame,
        first_column="first",
        second_column="second",
        iterations=200,
        confidence_level=0.95,
        seed=42,
    )
    assert observed == pytest.approx(-1.0)
    assert lower == pytest.approx(-1.0)
    assert upper == pytest.approx(-1.0)
    assert p_value < 0.02
    assert len(samples) == 200


def test_compare_conditions_writes_pairwise_and_error_analysis_outputs(tmp_path) -> None:
    import json

    from sclc.config import AppConfig
    from sclc.evaluation.statistics import compare_conditions
    from sclc.options import EmbeddingModel
    from sclc.paths import evaluation_dir

    config = AppConfig.model_validate(
        {
            "project": {"seed": 7},
            "paths": {
                "processed_dir": tmp_path / "processed",
                "profile_dir": tmp_path / "profiles",
                "subset_dir": tmp_path / "subsets",
                "retrieval_unit_dir": tmp_path / "retrieval_units",
                "encoding_dir": tmp_path / "encodings",
                "ranking_dir": tmp_path / "rankings",
                "evaluation_dir": tmp_path / "evaluation",
                "hf_cache_dir": tmp_path / "cache",
            },
            "dataset": {"repo_id": "allenai/qasper"},
            "document": {},
            "chunking": {"canonical_tokenizer": "granite", "overlap_tokens": 0},
            "models": {
                "granite": {"model_id": "granite", "max_document_tokens": 32768},
                "jina": {"model_id": "jina", "max_document_tokens": 8192},
            },
            "sampling": {},
            "evaluation": {
                "bootstrap_iterations": 50,
                "bootstrap_metrics": ["ndcg_at_10"],
                "primary_metric": "ndcg_at_10",
                "error_analysis_per_direction": 1,
            },
        }
    )

    condition_scores = {
        "bm25": 0.0,
        "fixed_dense": 0.1,
        "section_isolated": 0.2,
        "section_constrained": 0.4,
        "length_matched": 0.25,
        "global": 0.3,
    }

    def write_evaluation(condition: str, model: str | None) -> None:
        directory = evaluation_dir(config, 512) / condition
        if model is not None:
            directory = directory / model
        directory.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "query_id": "q1",
                "document_id": "d1",
                "question": "Core question",
                "query_type": "factual",
                "analysis_set": "cross_model_core",
                "split": "test",
                "ndcg_at_10": condition_scores[condition],
            }
        ]
        if model != "jina":
            rows.append(
                {
                    "query_id": "q2",
                    "document_id": "d2",
                    "question": "Extended question",
                    "query_type": "synthesis",
                    "analysis_set": "granite_extended",
                    "split": "test",
                    "ndcg_at_10": condition_scores[condition] + 0.05,
                }
            )
        pd.DataFrame(rows).to_csv(directory / "query_metrics.csv", index=False)
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "configuration_fingerprint": f"{condition}-{model}",
                    "files": {"query_metrics": "query_metrics.csv"},
                }
            )
        )

    write_evaluation("bm25", None)
    for model in ("granite", "jina"):
        for condition in (
            "fixed_dense",
            "section_isolated",
            "section_constrained",
            "length_matched",
            "global",
        ):
            write_evaluation(condition, model)

    manifest = compare_conditions(config, chunk_size=512)
    output_dir = evaluation_dir(config, 512) / "comparisons"
    comparisons = pd.read_csv(output_dir / manifest["files"]["comparisons"])
    candidates = pd.read_csv(output_dir / manifest["files"]["error_analysis_candidates"])
    assert not comparisons.empty
    assert not candidates.empty
    assert set(comparisons["first_condition"]).issuperset(
        {"bm25", "section_isolated", "section_constrained", "global"}
    )
    assert "mean_difference_first_minus_second" in comparisons.columns
    assert list((output_dir / "bootstrap").glob("*.npz"))

    # Granite-only confirmatory analysis must not require Jina outputs.
    import shutil

    for condition in (
        "fixed_dense",
        "section_isolated",
        "section_constrained",
        "global",
    ):
        shutil.rmtree(evaluation_dir(config, 512) / condition / "jina")

    granite_manifest = compare_conditions(
        config, chunk_size=512, models=(EmbeddingModel.GRANITE,)
    )
    granite_output_dir = evaluation_dir(config, 512) / "comparisons" / "granite"
    granite_comparisons = pd.read_csv(
        granite_output_dir / granite_manifest["files"]["comparisons"]
    )
    assert set(granite_comparisons["model_key"]) == {"granite"}


def test_uncertain_queries_are_excluded_from_category_slices() -> None:
    from sclc.evaluation.statistics import _group_slices

    frame = pd.DataFrame(
        {
            "query_type": ["factual", "uncertain", "unclassified"],
            "document_id": ["d1", "d2", "d3"],
        }
    )
    slices = _group_slices(frame)
    labels = {(group_type, group_value) for group_type, group_value, _ in slices}
    assert ("overall", "all") in labels
    assert ("query_type", "factual") in labels
    assert ("query_type", "uncertain") not in labels
    assert ("query_type", "unclassified") not in labels
