from __future__ import annotations

from sclc.config import AppConfig
from sclc.encoding.bm25 import build_bm25_encoding
from sclc.encoding.dense import build_dense_encoding
from sclc.options import EmbeddingModel, RetrievalCondition


def encode_condition(
    config: AppConfig,
    *,
    condition: RetrievalCondition,
    model: EmbeddingModel | None,
    chunk_size: int,
    overwrite: bool,
) -> dict[str, object]:
    """Dispatch one of the five controlled representation conditions."""
    match condition:
        case RetrievalCondition.BM25:
            if model is not None:
                raise ValueError("--model must not be supplied for the BM25 condition")
            return build_bm25_encoding(
                config,
                chunk_size=chunk_size,
                overwrite=overwrite,
            )
        case (
            RetrievalCondition.FIXED_DENSE
            | RetrievalCondition.SECTION_ISOLATED
            | RetrievalCondition.SECTION_CONSTRAINED
            | RetrievalCondition.GLOBAL
            | RetrievalCondition.LENGTH_MATCHED
        ):
            if model is None:
                raise ValueError(f"--model is required for {condition.value}")
            return build_dense_encoding(
                config,
                condition=condition,
                model_key=model,
                chunk_size=chunk_size,
                overwrite=overwrite,
            )
        case _:
            raise ValueError(f"Unsupported retrieval condition: {condition!r}")
