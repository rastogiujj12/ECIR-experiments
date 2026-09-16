import json
from pathlib import Path

import pandas as pd

from sclc.analysis.cross_section_challenge_analysis import analyse_cross_section_challenge
from sclc.config import AppConfig
from sclc.options import EmbeddingModel, RetrievalCondition
from sclc.paths import evaluation_dir

METRICS = [
    "ndcg_at_5",
    "recall_at_5",
    "evidence_paragraph_recall_at_5",
    "complete_evidence_at_5",
    "evidence_paragraph_recall_at_token_budget_1024",
    "complete_evidence_at_token_budget_2048",
]


def make_config(tmp_path: Path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "project": {"seed": 42},
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
                "supported_chunk_sizes": [128, 256],
            },
            "models": {
                "granite": {"model_id": "granite", "max_document_tokens": 32768},
                "jina": {"model_id": "jina", "max_document_tokens": 8192},
            },
            "sampling": {},
            "evaluation": {
                "cutoffs": [1, 3, 5, 10],
                "token_budgets": [512, 1024, 2048, 4096],
                "primary_metric": "ndcg_at_5",
                "bootstrap_iterations": 100,
                "confidence_level": 0.95,
                "bootstrap_metrics": METRICS,
                "require_query_types": False,
            },
        }
    )


def write_evaluation(
    config: AppConfig,
    *,
    size: int,
    condition: RetrievalCondition,
    model: EmbeddingModel,
    rows: list[dict[str, object]],
) -> None:
    directory = evaluation_dir(config, size) / condition.value / model.value
    directory.mkdir(parents=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(directory / "query_metrics.csv", index=False)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "configuration_fingerprint": f"{size}-{condition.value}-{model.value}",
                "files": {"query_metrics": "query_metrics.csv"},
            }
        ),
        encoding="utf-8",
    )


def test_cross_section_analysis_filters_accepted_queries(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    accepted_path = tmp_path / "accepted_queries.csv"
    accepted = pd.DataFrame(
        {
            "query_id": ["q1", "q2", "q3"],
            "document_id": ["d1", "d1", "d2"],
            "eligibility_group": [
                "cross_model_core",
                "cross_model_core",
                "granite_extended",
            ],
            "include": ["yes", "yes", "yes"],
        }
    )
    accepted.to_csv(accepted_path, index=False)

    for size in (128, 256):
        for condition_index, condition in enumerate(
            (
                RetrievalCondition.SECTION_ISOLATED,
                RetrievalCondition.SECTION_CONSTRAINED,
                RetrievalCondition.GLOBAL,
            )
        ):
            rows = []
            for query_index, (query_id, document_id, analysis_set) in enumerate(
                [
                    ("q1", "d1", "cross_model_core"),
                    ("q2", "d1", "cross_model_core"),
                    ("q3", "d2", "granite_extended"),
                    ("other", "d3", "cross_model_core"),
                ]
            ):
                base = 0.1 * query_index + 0.05 * condition_index
                row = {
                    "query_id": query_id,
                    "document_id": document_id,
                    "question": f"Question {query_id}",
                    "query_type": "unclassified",
                    "analysis_set": analysis_set,
                    "split": "test",
                }
                for metric in METRICS:
                    row[metric] = base
                rows.append(row)
            write_evaluation(
                config,
                size=size,
                condition=condition,
                model=EmbeddingModel.GRANITE,
                rows=rows,
            )

    manifest = analyse_cross_section_challenge(
        config,
        accepted_queries_path=accepted_path,
        model=EmbeddingModel.GRANITE,
        chunk_sizes=[128, 256],
    )
    assert manifest["kind"] == "cross_section_challenge_analysis"
    output_dir = (
        config.paths.analysis_dir
        / "cross_section_challenge_results"
        / "granite"
    )
    summary = pd.read_csv(output_dir / "summary_by_retrieval_unit_size.csv")
    comparisons = pd.read_csv(output_dir / "comparisons_within_retrieval_unit_size.csv")
    assert set(summary["analysis_group"]) == {
        "accepted_all",
        "accepted_cross_model",
    }
    assert set(summary.loc[summary["analysis_group"] == "accepted_all", "query_count"]) == {3}
    assert set(
        summary.loc[summary["analysis_group"] == "accepted_cross_model", "query_count"]
    ) == {2}
    assert len(comparisons) == 2 * 2 * 2 * len(METRICS)
    assert "mean_difference_first_minus_second" in comparisons.columns
