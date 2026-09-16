from __future__ import annotations

from pathlib import Path

from sclc.config import AppConfig


def resolve_retrieval_unit_size(config: AppConfig, retrieval_unit_size: int | None) -> int:
    """Resolve and validate a retrieval-unit size for a size-dependent stage."""
    resolved = (
        config.chunking.chunk_size_tokens
        if retrieval_unit_size is None
        else int(retrieval_unit_size)
    )
    if resolved <= 0:
        raise ValueError("--retrieval-unit-size must be a positive integer")
    supported = set(config.chunking.supported_chunk_sizes)
    if supported and resolved not in supported:
        raise ValueError(
            f"Unsupported retrieval-unit size {resolved}; expected one of {sorted(supported)}"
        )
    return resolved


def chunk_namespace(retrieval_unit_size: int) -> str:
    """Return the historical on-disk namespace used for one retrieval-unit size."""
    return f"chunk_{retrieval_unit_size}"


def retrieval_unit_dir(config: AppConfig, retrieval_unit_size: int) -> Path:
    return config.paths.retrieval_unit_dir / chunk_namespace(retrieval_unit_size)


def encoding_dir(config: AppConfig, retrieval_unit_size: int) -> Path:
    return config.paths.encoding_dir / chunk_namespace(retrieval_unit_size)


def ranking_dir(config: AppConfig, retrieval_unit_size: int) -> Path:
    return config.paths.ranking_dir / chunk_namespace(retrieval_unit_size)


def evaluation_dir(config: AppConfig, retrieval_unit_size: int) -> Path:
    return config.paths.evaluation_dir / chunk_namespace(retrieval_unit_size)


def analysis_dir(config: AppConfig, retrieval_unit_size: int) -> Path:
    return config.paths.analysis_dir / chunk_namespace(retrieval_unit_size)


def global_query_types_path(config: AppConfig) -> Path:
    return config.paths.retrieval_unit_dir / config.evaluation.query_types_filename


def global_query_coding_path(config: AppConfig) -> Path:
    return config.paths.retrieval_unit_dir / "query_type_coding.csv"
