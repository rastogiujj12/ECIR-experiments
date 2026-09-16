import json
import warnings
from pathlib import Path

import pandas as pd
from pandas.errors import PerformanceWarning

from sclc.config import AppConfig
from sclc.data.io import write_documents_jsonl
from sclc.data.retrieval_unit_io import write_models_jsonl
from sclc.data.schema import (
    CharacterSpan,
    DocumentRecord,
    EvidenceSetRecord,
    ParagraphRecord,
    PreparedQueryRecord,
    RankingRecord,
    RetrievalUnitRecord,
)
from sclc.evaluation.metrics import evaluate_condition
from sclc.options import RetrievalCondition
from sclc.paths import evaluation_dir, ranking_dir, retrieval_unit_dir


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
            "ranking": {"max_depth": 10},
            "evaluation": {"cutoffs": [1, 3, 5, 10]},
        }
    )


def test_alternative_evidence_and_union_metrics_are_both_retained(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    document = DocumentRecord(
        document_id="paper-1",
        split="test",
        title="Paper",
        abstract="",
        text="abcdefghijABCDEFGHIJ",
        sections=[],
        paragraphs=[
            ParagraphRecord(
                paragraph_id="p1",
                section_id="s1",
                source_section_index=0,
                source_paragraph_index=0,
                text="abcdefghij",
                span=CharacterSpan(start=0, end=10),
            ),
            ParagraphRecord(
                paragraph_id="p2",
                section_id="s1",
                source_section_index=0,
                source_paragraph_index=1,
                text="ABCDEFGHIJ",
                span=CharacterSpan(start=10, end=20),
            ),
        ],
        queries=[],
    )
    write_documents_jsonl([document], config.paths.processed_dir / "documents.jsonl")
    units = [
        RetrievalUnitRecord(
            retrieval_unit_id="u1",
            document_id="paper-1",
            analysis_set="cross_model_core",
            segmentation_plan="continuous",
            unit_index=0,
            span=CharacterSpan(start=0, end=10),
            text="abcdefghij",
            token_count=1,
            scope_token_start=0,
            scope_token_end=1,
            overlapping_paragraph_ids=["p1"],
        ),
        RetrievalUnitRecord(
            retrieval_unit_id="u2",
            document_id="paper-1",
            analysis_set="cross_model_core",
            segmentation_plan="continuous",
            unit_index=1,
            span=CharacterSpan(start=10, end=20),
            text="ABCDEFGHIJ",
            token_count=1,
            scope_token_start=1,
            scope_token_end=2,
            overlapping_paragraph_ids=["p2"],
        ),
    ]
    write_models_jsonl(units, retrieval_unit_dir(config, 512) / "continuous_units.jsonl")
    query = PreparedQueryRecord(
        query_id="q1",
        document_id="paper-1",
        split="test",
        analysis_set="cross_model_core",
        question="Which evidence?",
        evidence_union_paragraph_ids=["p1", "p2"],
        evidence_sets=[
            EvidenceSetRecord(evidence_set_id="e1", paragraph_ids=["p1"]),
            EvidenceSetRecord(evidence_set_id="e2", paragraph_ids=["p2"]),
        ],
    )
    write_models_jsonl([query], retrieval_unit_dir(config, 512) / "queries.jsonl")

    ranking_output_dir = ranking_dir(config, 512) / "bm25"
    ranking_output_dir.mkdir(parents=True)
    write_models_jsonl(
        [
            RankingRecord(
                query_id="q1",
                document_id="paper-1",
                analysis_set="cross_model_core",
                condition="bm25",
                segmentation_plan="continuous",
                retrieval_unit_id="u2",
                rank=1,
                score=2.0,
                character_start=10,
                character_end=20,
                overlapping_paragraph_ids=["p2"],
            ),
            RankingRecord(
                query_id="q1",
                document_id="paper-1",
                analysis_set="cross_model_core",
                condition="bm25",
                segmentation_plan="continuous",
                retrieval_unit_id="u1",
                rank=2,
                score=1.0,
                character_start=0,
                character_end=10,
                overlapping_paragraph_ids=["p1"],
            ),
        ],
        ranking_output_dir / "rankings.jsonl",
    )
    (ranking_output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "configuration_fingerprint": "ranking-fingerprint",
                "file": "rankings.jsonl",
            }
        )
    )
    pd.DataFrame([{"query_id": "q1", "query_type": "factual"}]).to_csv(
        config.paths.retrieval_unit_dir / "query_types.csv", index=False
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        evaluate_condition(
            config,
            condition=RetrievalCondition.BM25,
            model=None,
            chunk_size=512,
        )
    frame = pd.read_csv(evaluation_dir(config, 512) / "bm25" / "query_metrics.csv")
    row = frame.iloc[0]
    # Primary ranking and coverage metrics use the evidence union.
    assert row["precision_at_1"] == 1.0
    assert row["recall_at_1"] == 0.5
    assert row["evidence_paragraph_recall_at_1"] == 0.5
    assert row["evidence_span_coverage_at_1"] == 0.5
    # Complete support succeeds when any acceptable evidence set is complete.
    assert row["complete_evidence_at_1"] == 1.0
    # Best-set values remain available as alternative-evidence outputs.
    assert row["best_recall_at_1"] == 1.0
    assert row["best_complete_evidence_at_1"] == 1.0
    assert row["union_complete_evidence_at_1"] == 0.0


def test_evidence_span_coverage_ignores_separator_whitespace() -> None:
    from sclc.evaluation.metrics import evidence_span_coverage_at_k

    unit = RetrievalUnitRecord(
        retrieval_unit_id="u",
        document_id="d",
        analysis_set="cross_model_core",
        segmentation_plan="continuous",
        unit_index=0,
        span=CharacterSpan(start=0, end=5),
        text="alpha",
        token_count=1,
        scope_token_start=0,
        scope_token_end=1,
        overlapping_paragraph_ids=["p"],
    )
    # The paragraph includes a trailing separator, but all content characters are covered.
    score = evidence_span_coverage_at_k(
        ["u"],
        {"u": unit},
        {"p": (0, 8)},
        {"p"},
        1,
        "alpha   ",
    )
    assert score == 1.0


def test_evaluation_requires_complete_query_type_coding(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.paths.retrieval_unit_dir.mkdir(parents=True)
    pd.DataFrame([{"query_id": "other", "query_type": "uncertain"}]).to_csv(
        config.paths.retrieval_unit_dir / "query_types.csv", index=False
    )
    import pytest

    from sclc.data.query_types import load_query_types

    with pytest.raises(ValueError, match="missing 1 required query IDs"):
        load_query_types(config, {"q1"})
