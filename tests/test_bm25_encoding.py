import gzip
import json
from pathlib import Path

from sclc.config import AppConfig
from sclc.data.retrieval_unit_io import write_models_jsonl
from sclc.data.schema import CharacterSpan, RetrievalUnitRecord
from sclc.encoding.bm25 import build_bm25_encoding, lexical_tokens
from sclc.paths import encoding_dir, retrieval_unit_dir


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
            "chunking": {
                "canonical_tokenizer": "granite",
                "chunk_size_tokens": 512,
                "overlap_tokens": 0,
            },
            "models": {
                "granite": {"model_id": "granite", "max_document_tokens": 32768},
                "jina": {"model_id": "jina", "max_document_tokens": 8192},
            },
            "sampling": {},
            "bm25": {
                "lowercase": True,
                "token_pattern": r"(?u)\b\w[\w'-]*\b",
                "k1": 1.5,
                "b": 0.75,
            },
        }
    )


def make_unit(unit_id: str, index: int, text: str) -> RetrievalUnitRecord:
    return RetrievalUnitRecord(
        retrieval_unit_id=unit_id,
        document_id="1610.06510",
        analysis_set="cross_model_core",
        segmentation_plan="continuous",
        unit_index=index,
        span=CharacterSpan(start=index * 10, end=index * 10 + len(text)),
        text=text,
        token_count=3,
        scope_token_start=index * 3,
        scope_token_end=index * 3 + 3,
    )


def test_lexical_tokenisation_is_lowercase_without_stemming() -> None:
    tokens = lexical_tokens(
        "Retrieval-based systems don't rewrite terms.",
        pattern=r"(?u)\b\w[\w'-]*\b",
        lowercase=True,
    )
    assert tokens == ["retrieval-based", "systems", "don't", "rewrite", "terms"]


def test_bm25_encoding_writes_resumable_document_index(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    units_path = retrieval_unit_dir(config, 512) / "continuous_units.jsonl"
    write_models_jsonl(
        [
            make_unit("u1", 0, "Dense retrieval retrieval"),
            make_unit("u2", 1, "Lexical retrieval"),
        ],
        units_path,
    )

    manifest = build_bm25_encoding(config, chunk_size=512)
    assert manifest["document_count"] == 1
    assert manifest["unit_count"] == 2

    document_file = encoding_dir(config, 512) / "bm25" / manifest["documents"][0]["file"]
    with gzip.open(document_file, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["document_id"] == "1610.06510"
    assert payload["document_frequency"]["retrieval"] == 2
    assert payload["units"][0]["term_frequency"]["retrieval"] == 2

    cached = build_bm25_encoding(config, chunk_size=512)
    assert cached["cached_document_count"] == 1
