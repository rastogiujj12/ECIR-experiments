from pathlib import Path

import pytest

from sclc.config import AppConfig
from sclc.data.schema import CharacterSpan, RetrievalUnitRecord
from sclc.evaluation.metrics import (
    _prefix_for_token_budget,
    average_precision,
    first_relevant_rank,
    r_precision,
    reciprocal_rank,
)
from sclc.paths import (
    encoding_dir,
    evaluation_dir,
    ranking_dir,
    resolve_retrieval_unit_size,
    retrieval_unit_dir,
)


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
                "chunk_size_tokens": 512,
                "supported_chunk_sizes": [128, 256, 512],
            },
            "models": {
                "granite": {"model_id": "granite", "max_document_tokens": 32768},
                "jina": {"model_id": "jina", "max_document_tokens": 8192},
            },
            "sampling": {},
        }
    )


def test_chunk_dependent_outputs_are_namespaced(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    assert retrieval_unit_dir(config, 128).name == "chunk_128"
    assert encoding_dir(config, 256).name == "chunk_256"
    assert ranking_dir(config, 512).name == "chunk_512"
    assert evaluation_dir(config, 128).name == "chunk_128"
    assert resolve_retrieval_unit_size(config, None) == 512
    assert resolve_retrieval_unit_size(config, 256) == 256
    with pytest.raises(ValueError, match="expected one of"):
        resolve_retrieval_unit_size(config, 300)


def test_full_ranking_metrics() -> None:
    ranked = ["a", "b", "c", "d"]
    relevant = {"b", "d"}
    assert average_precision(ranked, relevant) == pytest.approx(0.5)
    assert reciprocal_rank(ranked, relevant) == pytest.approx(0.5)
    assert r_precision(ranked, relevant) == pytest.approx(0.5)
    assert first_relevant_rank(ranked, relevant) == 2


def test_token_budget_uses_a_ranked_prefix_without_skipping() -> None:
    units = {
        unit_id: RetrievalUnitRecord(
            retrieval_unit_id=unit_id,
            document_id="d",
            analysis_set="cross_model_core",
            segmentation_plan="continuous",
            unit_index=index,
            span=CharacterSpan(start=index * 10, end=index * 10 + 5),
            text=unit_id,
            token_count=tokens,
            scope_token_start=index,
            scope_token_end=index + 1,
        )
        for index, (unit_id, tokens) in enumerate(
            [("a", 128), ("b", 128), ("c", 256), ("d", 32)]
        )
    }
    prefix, tokens = _prefix_for_token_budget(["a", "b", "c", "d"], units, 300)
    # The third ranked unit exceeds the remaining budget. Later units are not
    # skipped because token-budget retrieval must remain a ranking prefix.
    assert prefix == ["a", "b"]
    assert tokens == 256
