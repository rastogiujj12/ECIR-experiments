import json
from pathlib import Path

import numpy as np

from sclc.config import AppConfig
from sclc.data.retrieval_unit_io import write_models_jsonl
from sclc.data.schema import (
    CharacterSpan,
    EvidenceSetRecord,
    PreparedQueryRecord,
    RetrievalUnitRecord,
)
from sclc.encoding.bm25 import build_bm25_encoding
from sclc.options import EmbeddingModel, RetrievalCondition
from sclc.paths import encoding_dir, ranking_dir, retrieval_unit_dir
from sclc.retrieval.ranking import _stable_order, rank_condition, read_rankings


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
        }
    )


def make_inputs(config: AppConfig) -> None:
    units = [
        RetrievalUnitRecord(
            retrieval_unit_id="u-b",
            document_id="paper-1",
            analysis_set="cross_model_core",
            segmentation_plan="continuous",
            unit_index=0,
            span=CharacterSpan(start=0, end=10),
            text="alpha alpha",
            token_count=2,
            scope_token_start=0,
            scope_token_end=2,
            overlapping_paragraph_ids=["p1"],
        ),
        RetrievalUnitRecord(
            retrieval_unit_id="u-a",
            document_id="paper-1",
            analysis_set="cross_model_core",
            segmentation_plan="continuous",
            unit_index=1,
            span=CharacterSpan(start=11, end=21),
            text="beta beta",
            token_count=2,
            scope_token_start=2,
            scope_token_end=4,
            overlapping_paragraph_ids=["p2"],
        ),
    ]
    write_models_jsonl(units, retrieval_unit_dir(config, 512) / "continuous_units.jsonl")
    write_models_jsonl(
        [
            PreparedQueryRecord(
                query_id="q1",
                document_id="paper-1",
                split="test",
                analysis_set="cross_model_core",
                question="alpha",
                evidence_union_paragraph_ids=["p1"],
                evidence_sets=[EvidenceSetRecord(evidence_set_id="e1", paragraph_ids=["p1"])],
            )
        ],
        retrieval_unit_dir(config, 512) / "queries.jsonl",
    )


def test_stable_order_uses_unit_id_for_ties() -> None:
    assert _stable_order(np.asarray([1.0, 1.0]), ["u-b", "u-a"]) == [1, 0]


def test_bm25_retrieval_ranks_within_document(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    make_inputs(config)
    build_bm25_encoding(config, chunk_size=512)
    manifest = rank_condition(config, condition=RetrievalCondition.BM25, model=None, chunk_size=512)
    rankings = read_rankings(ranking_dir(config, 512) / "bm25" / manifest["file"])
    assert [record.retrieval_unit_id for record in rankings] == ["u-b", "u-a"]
    assert rankings[0].score > rankings[1].score


def test_dense_retrieval_uses_shared_query_vector(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    make_inputs(config)
    passage_dir = encoding_dir(config, 512) / "fixed_dense" / "granite"
    document_dir = passage_dir / "documents"
    document_dir.mkdir(parents=True)
    np.savez_compressed(
        document_dir / "paper.npz",
        embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        retrieval_unit_ids=np.asarray(["u-b", "u-a"]),
        unit_indices=np.asarray([0, 1], dtype=np.int32),
    )
    (passage_dir / "manifest.json").write_text(
        json.dumps(
            {
                "configuration_fingerprint": "passage-fingerprint",
                "documents": [
                    {
                        "document_id": "paper-1",
                        "file": "documents/paper.npz",
                        "unit_count": 2,
                    }
                ],
            }
        )
    )
    query_dir = config.paths.encoding_dir / "queries" / "granite"
    query_dir.mkdir(parents=True)
    np.savez_compressed(
        query_dir / "queries.npz",
        embeddings=np.asarray([[0.0, 1.0]], dtype=np.float32),
        query_ids=np.asarray(["q1"]),
        document_ids=np.asarray(["paper-1"]),
    )
    (query_dir / "manifest.json").write_text(
        json.dumps(
            {
                "configuration_fingerprint": "query-fingerprint",
                "file": "queries.npz",
            }
        )
    )

    manifest = rank_condition(
        config,
        condition=RetrievalCondition.FIXED_DENSE,
        model=EmbeddingModel.GRANITE,
        chunk_size=512,
    )
    rankings = read_rankings(
        ranking_dir(config, 512) / "fixed_dense" / "granite" / manifest["file"]
    )
    assert [record.retrieval_unit_id for record in rankings] == ["u-a", "u-b"]


def test_units_fingerprint_changes_when_evidence_overlap_changes() -> None:
    from sclc.data.schema import CharacterSpan, RetrievalUnitRecord
    from sclc.retrieval.ranking import _units_fingerprint

    base = RetrievalUnitRecord(
        retrieval_unit_id="u1",
        document_id="d1",
        analysis_set="cross_model_core",
        segmentation_plan="continuous",
        unit_index=0,
        span=CharacterSpan(start=0, end=5),
        text="alpha",
        token_count=1,
        scope_token_start=0,
        scope_token_end=1,
        overlapping_paragraph_ids=["p1"],
    )
    changed = base.model_copy(update={"overlapping_paragraph_ids": ["p2"]})
    assert _units_fingerprint([base]) != _units_fingerprint([changed])
