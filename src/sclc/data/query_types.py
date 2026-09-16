from __future__ import annotations

import pandas as pd

from sclc.config import AppConfig
from sclc.data.retrieval_unit_io import read_prepared_queries
from sclc.options import EmbeddingModel
from sclc.paths import global_query_types_path, retrieval_unit_dir

ALLOWED_QUERY_TYPES = {
    "factual",
    "section_specific",
    "multi_hop",
    "synthesis",
    "uncertain",
}


def expected_query_ids(
    config: AppConfig,
    model: EmbeddingModel | None,
    *,
    chunk_size: int,
) -> set[str]:
    path = retrieval_unit_dir(config, chunk_size) / "queries.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run `sclc build-units "
            f"--retrieval-unit-size {chunk_size}` first."
        )
    return {
        query.query_id
        for query in read_prepared_queries(path)
        if model is not EmbeddingModel.JINA
        or query.analysis_set == "cross_model_core"
    }


def load_query_types(
    config: AppConfig,
    expected_ids: set[str],
) -> dict[str, str]:
    # Query coding is independent of retrieval-unit size and therefore lives once at the
    # retrieval-unit root instead of being duplicated across size namespaces.
    path = global_query_types_path(config)
    if not path.exists():
        if config.evaluation.require_query_types:
            raise FileNotFoundError(
                f"{path} does not exist. Complete the query-type coding sheet before "
                "running retrieval. Copy query_type_coding.csv to "
                f"{config.evaluation.query_types_filename} and assign every query to "
                f"one of {sorted(ALLOWED_QUERY_TYPES)}."
            )
        return {}

    frame = pd.read_csv(
        path,
        dtype={"query_id": "string", "query_type": "string"},
        keep_default_na=False,
    )
    required = {"query_id", "query_type"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    duplicates = frame[frame["query_id"].duplicated()]["query_id"].tolist()
    if duplicates:
        raise ValueError(f"{path} contains duplicate query IDs: {duplicates[:10]}")

    mapping: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        query_id = str(row.query_id).strip()
        query_type = (
            str(row.query_type).strip().lower().replace("-", "_").replace(" ", "_")
        )
        if not query_id:
            raise ValueError(f"{path} contains a blank query_id")
        if query_type not in ALLOWED_QUERY_TYPES:
            raise ValueError(
                f"Unsupported query type {query_type!r} for {query_id}; "
                f"expected one of {sorted(ALLOWED_QUERY_TYPES)}"
            )
        mapping[query_id] = query_type

    if config.evaluation.require_query_types:
        missing_ids = sorted(expected_ids.difference(mapping))
        if missing_ids:
            raise ValueError(
                f"{path} is missing {len(missing_ids)} required query IDs "
                f"(first: {missing_ids[:10]})."
            )
    return {query_id: mapping[query_id] for query_id in expected_ids if query_id in mapping}


def validate_query_type_coding(
    config: AppConfig,
    model: EmbeddingModel | None,
    *,
    chunk_size: int,
) -> dict[str, str]:
    return load_query_types(
        config,
        expected_query_ids(config, model, chunk_size=chunk_size),
    )


__all__ = [
    "ALLOWED_QUERY_TYPES",
    "expected_query_ids",
    "load_query_types",
    "validate_query_type_coding",
]
