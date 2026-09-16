from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from sclc.config import AppConfig
from sclc.data.retrieval_unit_io import read_prepared_queries, read_retrieval_units
from sclc.data.schema import PreparedQueryRecord, RankingRecord, RetrievalUnitRecord
from sclc.encoding.bm25 import lexical_tokens
from sclc.encoding.dense import condition_spec
from sclc.options import EmbeddingModel, RetrievalCondition
from sclc.paths import encoding_dir, ranking_dir, retrieval_unit_dir


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run the preceding pipeline stage first.")
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()



def _units_fingerprint(units: Sequence[RetrievalUnitRecord]) -> str:
    return _fingerprint(
        {
            "units": [
                {
                    "retrieval_unit_id": unit.retrieval_unit_id,
                    "document_id": unit.document_id,
                    "analysis_set": unit.analysis_set,
                    "segmentation_plan": unit.segmentation_plan,
                    "span": [unit.span.start, unit.span.end],
                    "overlapping_paragraph_ids": unit.overlapping_paragraph_ids,
                }
                for unit in sorted(
                    units,
                    key=lambda item: (
                        item.document_id,
                        item.unit_index,
                        item.retrieval_unit_id,
                    ),
                )
            ]
        }
    )

def _output_dir(
    config: AppConfig,
    condition: RetrievalCondition,
    model: EmbeddingModel | None,
    *,
    chunk_size: int,
) -> Path:
    base = ranking_dir(config, chunk_size)
    if condition is RetrievalCondition.BM25:
        return base / condition.value
    assert model is not None
    return base / condition.value / model.value


def _write_rankings(records: Iterable[RankingRecord], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json())
            handle.write("\n")
            count += 1
    temporary.replace(path)
    return count


def read_rankings(path: Path) -> list[RankingRecord]:
    records: list[RankingRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(RankingRecord.model_validate_json(line))
    return records


def _load_queries(
    config: AppConfig,
    *,
    chunk_size: int,
) -> list[PreparedQueryRecord]:
    path = retrieval_unit_dir(config, chunk_size) / "queries.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run `sclc build-units` first.")
    return list(read_prepared_queries(path))


def _units_for_condition(
    config: AppConfig,
    condition: RetrievalCondition,
    *,
    chunk_size: int,
) -> list[RetrievalUnitRecord]:
    filename = (
        "continuous_units.jsonl"
        if condition in {RetrievalCondition.BM25, RetrievalCondition.FIXED_DENSE}
        else "section_bounded_units.jsonl"
    )
    path = retrieval_unit_dir(config, chunk_size) / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run `sclc build-units` first.")
    return list(read_retrieval_units(path))


def _eligible_analysis_set(model: EmbeddingModel | None, analysis_set: str) -> bool:
    return model is None or model is EmbeddingModel.GRANITE or analysis_set == "cross_model_core"


def _stable_order(scores: np.ndarray, unit_ids: Sequence[str]) -> list[int]:
    if scores.ndim != 1 or len(scores) != len(unit_ids):
        raise ValueError("Scores and retrieval-unit identifiers have incompatible shapes")
    return sorted(range(len(unit_ids)), key=lambda index: (-float(scores[index]), unit_ids[index]))


def _ranking_records(
    *,
    query: PreparedQueryRecord,
    condition: RetrievalCondition,
    model: EmbeddingModel | None,
    units_by_id: dict[str, RetrievalUnitRecord],
    unit_ids: Sequence[str],
    scores: np.ndarray,
    max_depth: int,
) -> list[RankingRecord]:
    records: list[RankingRecord] = []
    for rank, index in enumerate(_stable_order(scores, unit_ids)[:max_depth], start=1):
        unit = units_by_id[unit_ids[index]]
        if unit.document_id != query.document_id:
            raise RuntimeError("Within-document retrieval received a unit from another paper")
        records.append(
            RankingRecord(
                query_id=query.query_id,
                document_id=query.document_id,
                analysis_set=query.analysis_set,
                condition=condition.value,
                model_key=model.value if model is not None else None,
                segmentation_plan=unit.segmentation_plan,
                retrieval_unit_id=unit.retrieval_unit_id,
                rank=rank,
                score=float(scores[index]),
                parent_section_id=unit.parent_section_id,
                character_start=unit.span.start,
                character_end=unit.span.end,
                overlapping_paragraph_ids=unit.overlapping_paragraph_ids,
            )
        )
    return records


def _bm25_scores(
    *,
    query_text: str,
    payload: dict[str, Any],
) -> tuple[list[str], np.ndarray]:
    query_terms = lexical_tokens(
        query_text,
        pattern=str(payload["token_pattern"]),
        lowercase=bool(payload["lowercase"]),
    )
    unique_query_terms = sorted(set(query_terms))
    units = payload["units"]
    unit_ids = [str(unit["retrieval_unit_id"]) for unit in units]
    scores = np.zeros(len(units), dtype=np.float64)
    n_units = len(units)
    average_length = float(payload["average_document_length"])
    k1 = float(payload["k1"])
    b = float(payload["b"])
    document_frequency = payload["document_frequency"]

    if n_units == 0:
        return unit_ids, scores

    for term in unique_query_terms:
        df = int(document_frequency.get(term, 0))
        if df <= 0:
            continue
        idf = math.log(1.0 + (n_units - df + 0.5) / (df + 0.5))
        for index, unit in enumerate(units):
            tf = int(unit["term_frequency"].get(term, 0))
            if tf == 0:
                continue
            length = int(unit["length"])
            normalizer = 1.0 - b
            if average_length > 0:
                normalizer += b * length / average_length
            denominator = tf + k1 * normalizer
            scores[index] += idf * (tf * (k1 + 1.0)) / denominator
    return unit_ids, scores


def _rank_bm25(
    config: AppConfig,
    *,
    queries: Sequence[PreparedQueryRecord],
    units_by_id: dict[str, RetrievalUnitRecord],
    manifest: dict[str, Any],
    chunk_size: int,
) -> list[RankingRecord]:
    source_dir = encoding_dir(config, chunk_size) / "bm25"
    files = {
        str(item["document_id"]): source_dir / str(item["file"])
        for item in manifest["documents"]
    }
    records: list[RankingRecord] = []
    payload_cache: dict[str, dict[str, Any]] = {}
    for query in queries:
        path = files.get(query.document_id)
        if path is None:
            raise RuntimeError(f"BM25 encoding is missing paper {query.document_id}")
        payload = payload_cache.get(query.document_id)
        if payload is None:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            payload_cache[query.document_id] = payload
        unit_ids, scores = _bm25_scores(query_text=query.question, payload=payload)
        records.extend(
            _ranking_records(
                query=query,
                condition=RetrievalCondition.BM25,
                model=None,
                units_by_id=units_by_id,
                unit_ids=unit_ids,
                scores=scores,
                max_depth=(
                    len(unit_ids)
                    if config.ranking.store_complete_ranking
                    else config.ranking.max_depth
                ),
            )
        )
    return records


def _load_query_embeddings(
    config: AppConfig,
    model: EmbeddingModel,
) -> tuple[dict[str, tuple[np.ndarray, str]], dict[str, Any]]:
    directory = config.paths.encoding_dir / "queries" / model.value
    manifest = _read_json(directory / "manifest.json")
    path = directory / str(manifest["file"])
    with np.load(path, allow_pickle=False) as payload:
        embeddings = payload["embeddings"].astype(np.float32, copy=False)
        query_ids = payload["query_ids"].tolist()
        document_ids = payload["document_ids"].tolist()
    if len(query_ids) != len(set(query_ids)):
        raise RuntimeError("Dense query encoding contains duplicate query identifiers")
    mapping: dict[str, tuple[np.ndarray, str]] = {}
    for index, (query_id, document_id) in enumerate(
        zip(query_ids, document_ids, strict=True)
    ):
        if not str(document_id):
            raise RuntimeError(f"Query {query_id} has no associated document")
        mapping[str(query_id)] = (embeddings[index], str(document_id))
    return mapping, manifest


def _rank_dense(
    config: AppConfig,
    *,
    condition: RetrievalCondition,
    model: EmbeddingModel,
    queries: Sequence[PreparedQueryRecord],
    units_by_id: dict[str, RetrievalUnitRecord],
    manifest: dict[str, Any],
    chunk_size: int,
) -> tuple[list[RankingRecord], dict[str, Any]]:
    query_vectors, query_manifest = _load_query_embeddings(config, model)
    source_dir = encoding_dir(config, chunk_size) / condition.value / model.value
    files = {
        str(item["document_id"]): source_dir / str(item["file"])
        for item in manifest["documents"]
    }
    records: list[RankingRecord] = []
    passage_cache: dict[str, tuple[np.ndarray, list[str]]] = {}
    for query in queries:
        query_encoding = query_vectors.get(query.query_id)
        if query_encoding is None:
            raise RuntimeError(f"Dense query encoding is missing {query.query_id}")
        query_vector, encoded_document_id = query_encoding
        if encoded_document_id != query.document_id:
            raise RuntimeError(
                f"Dense query {query.query_id} is linked to {encoded_document_id}, "
                f"not {query.document_id}"
            )
        path = files.get(query.document_id)
        if path is None:
            raise RuntimeError(
                f"{condition.value}/{model.value} encoding is missing paper {query.document_id}"
            )
        cached_passages = passage_cache.get(query.document_id)
        if cached_passages is None:
            with np.load(path, allow_pickle=False) as payload:
                embeddings = payload["embeddings"].astype(np.float32, copy=False)
                unit_ids = [str(value) for value in payload["retrieval_unit_ids"].tolist()]
            passage_cache[query.document_id] = (embeddings, unit_ids)
        else:
            embeddings, unit_ids = cached_passages
        if embeddings.ndim != 2 or embeddings.shape[0] != len(unit_ids):
            raise RuntimeError(f"Invalid passage embedding file: {path}")
        if embeddings.shape[1] != query_vector.shape[0]:
            raise RuntimeError(f"Query and passage dimensions differ for {query.query_id}")
        scores = embeddings @ query_vector.astype(np.float32, copy=False)
        records.extend(
            _ranking_records(
                query=query,
                condition=condition,
                model=model,
                units_by_id=units_by_id,
                unit_ids=unit_ids,
                scores=scores,
                max_depth=(
                    len(unit_ids)
                    if config.ranking.store_complete_ranking
                    else config.ranking.max_depth
                ),
            )
        )
    return records, query_manifest


def rank_condition(
    config: AppConfig,
    *,
    condition: RetrievalCondition,
    model: EmbeddingModel | None,
    chunk_size: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    if condition is RetrievalCondition.BM25 and model is not None:
        raise ValueError("--model must not be supplied for BM25 retrieval")
    if condition.is_dense and model is None:
        raise ValueError(f"--model is required for {condition.value} retrieval")
    if (
        not config.ranking.store_complete_ranking
        and config.ranking.max_depth < max(config.evaluation.cutoffs)
    ):
        raise ValueError("ranking.max_depth must be at least the largest evaluation cutoff")

    output_dir = _output_dir(config, condition, model, chunk_size=chunk_size)
    rankings_path = output_dir / "rankings.jsonl"
    manifest_path = output_dir / "manifest.json"

    if condition is RetrievalCondition.BM25:
        source_dir = encoding_dir(config, chunk_size) / "bm25"
    else:
        assert model is not None
        source_dir = encoding_dir(config, chunk_size) / condition.value / model.value
    source_manifest = _read_json(source_dir / "manifest.json")

    queries = [
        query
        for query in _load_queries(config, chunk_size=chunk_size)
        if _eligible_analysis_set(model, query.analysis_set)
    ]
    units = [
        unit
        for unit in _units_for_condition(config, condition, chunk_size=chunk_size)
        if _eligible_analysis_set(model, unit.analysis_set)
    ]
    units_by_id = {unit.retrieval_unit_id: unit for unit in units}
    if len(units_by_id) != len(units):
        raise RuntimeError("Retrieval-unit identifiers are not unique")

    query_encoding_fingerprint = None
    if model is not None:
        query_encoding_manifest = _read_json(
            config.paths.encoding_dir / "queries" / model.value / "manifest.json"
        )
        query_encoding_fingerprint = query_encoding_manifest.get(
            "configuration_fingerprint"
        )
    query_payload = [
        {
            "query_id": query.query_id,
            "document_id": query.document_id,
            "analysis_set": query.analysis_set,
            "question": query.question,
        }
        for query in queries
    ]
    configuration = {
        "condition": condition.value,
        "chunk_size_tokens": chunk_size,
        "model_key": model.value if model is not None else None,
        "store_complete_ranking": config.ranking.store_complete_ranking,
        "max_depth": (
            "complete"
            if config.ranking.store_complete_ranking
            else config.ranking.max_depth
        ),
        "source_encoding_fingerprint": source_manifest.get("configuration_fingerprint"),
        "query_encoding_fingerprint": query_encoding_fingerprint,
        "queries_fingerprint": _fingerprint({"queries": query_payload}),
        "retrieval_units_fingerprint": _units_fingerprint(units),
    }
    fingerprint = _fingerprint(configuration)
    if rankings_path.exists() and manifest_path.exists() and not overwrite:
        existing = _read_json(manifest_path)
        if existing.get("configuration_fingerprint") != fingerprint:
            raise RuntimeError(
                f"Cached rankings at {rankings_path} do not match current inputs. "
                "Re-run with --overwrite."
            )
        return existing

    if condition is RetrievalCondition.BM25:
        records = _rank_bm25(
            config,
            queries=queries,
            units_by_id=units_by_id,
            manifest=source_manifest,
            chunk_size=chunk_size,
        )
        query_manifest: dict[str, Any] | None = None
        segmentation_plan = "continuous"
    else:
        assert model is not None
        records, query_manifest = _rank_dense(
            config,
            condition=condition,
            model=model,
            queries=queries,
            units_by_id=units_by_id,
            manifest=source_manifest,
            chunk_size=chunk_size,
        )
        segmentation_plan = condition_spec(condition).segmentation_plan

    records.sort(key=lambda item: (item.query_id, item.rank))
    result_count = _write_rankings(records, rankings_path)
    ranked_query_ids = {record.query_id for record in records}
    missing_queries = [
        query.query_id for query in queries if query.query_id not in ranked_query_ids
    ]
    if missing_queries:
        raise RuntimeError(f"No ranking was produced for queries: {missing_queries[:10]}")

    if config.ranking.store_complete_ranking:
        candidate_counts: dict[str, int] = {}
        for unit in units:
            candidate_counts[unit.document_id] = candidate_counts.get(unit.document_id, 0) + 1
        expected_results = sum(candidate_counts[query.document_id] for query in queries)
        if result_count != expected_results:
            raise RuntimeError(
                f"Complete ranking expected {expected_results} rows but wrote {result_count}"
            )

    manifest = {
        "schema_version": 1,
        "kind": "rankings",
        "condition": condition.value,
        "chunk_size_tokens": chunk_size,
        "model_key": model.value if model is not None else None,
        "segmentation_plan": segmentation_plan,
        "document_count": len({query.document_id for query in queries}),
        "query_count": len(queries),
        "result_count": result_count,
        "max_depth": (
            "complete"
            if config.ranking.store_complete_ranking
            else config.ranking.max_depth
        ),
        "store_complete_ranking": config.ranking.store_complete_ranking,
        "configuration": configuration,
        "configuration_fingerprint": fingerprint,
        "source_encoding_manifest": str(source_dir / "manifest.json"),
        "query_encoding_fingerprint": (
            query_encoding_fingerprint
        ),
        "file": rankings_path.name,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


__all__ = ["_bm25_scores", "_stable_order", "rank_condition", "read_rankings"]
