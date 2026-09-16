from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class ProjectConfig(BaseModel):
    seed: int = 42


class PathConfig(BaseModel):
    processed_dir: Path
    profile_dir: Path
    subset_dir: Path
    retrieval_unit_dir: Path = Path("data/retrieval_units")
    encoding_dir: Path = Path("outputs/encodings")
    ranking_dir: Path = Path("outputs/rankings")
    evaluation_dir: Path = Path("outputs/evaluation")
    analysis_dir: Path = Path("outputs/analysis")
    hf_cache_dir: Path


class DatasetConfig(BaseModel):
    repo_id: str
    subset: str | None = None
    splits: list[str] = Field(default_factory=lambda: ["train", "validation", "test"])
    parquet_files: dict[str, str] = Field(default_factory=dict)


class DocumentConfig(BaseModel):
    include_title: bool = True
    include_abstract: bool = True
    abstract_heading: str = "Abstract"
    hierarchy_separator: str = " ::: "
    heading_separator: str = "\n\n"
    paragraph_separator: str = "\n\n"
    section_separator: str = "\n\n"
    remove_reference_sections: bool = True
    reference_headings: list[str] = Field(
        default_factory=lambda: ["references", "bibliography", "works cited"]
    )


class ChunkingConfig(BaseModel):
    canonical_tokenizer: str
    canonical_tokenizer_revision: str | None = None
    tokenizer_trust_remote_code: bool = False
    chunk_size_tokens: int = 512
    supported_chunk_sizes: list[int] = Field(default_factory=lambda: [128, 256, 512])
    overlap_tokens: int = 0
    retain_short_final_chunk: bool = True

    @model_validator(mode="after")
    def validate_overlap(self) -> ChunkingConfig:
        if self.chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be positive")
        if not self.supported_chunk_sizes or any(
            size <= 0 for size in self.supported_chunk_sizes
        ):
            raise ValueError("supported_chunk_sizes must contain positive integers")
        if self.supported_chunk_sizes != sorted(set(self.supported_chunk_sizes)):
            raise ValueError("supported_chunk_sizes must be sorted and unique")
        if self.chunk_size_tokens not in self.supported_chunk_sizes:
            raise ValueError(
                "chunk_size_tokens must be included in supported_chunk_sizes"
            )
        if not 0 <= self.overlap_tokens < self.chunk_size_tokens:
            raise ValueError("overlap_tokens must be >= 0 and smaller than chunk_size_tokens")
        if self.overlap_tokens != 0:
            raise ValueError("This controlled experiment fixes overlap_tokens at 0")
        if not self.retain_short_final_chunk:
            raise ValueError("This experiment retains the shorter final chunk in every scope")
        return self


class ModelProfileConfig(BaseModel):
    model_id: str
    revision: str | None = None
    tokenizer_trust_remote_code: bool = False
    model_trust_remote_code: bool = False
    max_document_tokens: int
    passage_adapter: str | None = None
    query_adapter: str | None = None
    adapter_source: str | None = None
    adapter_revision: str | None = None


class ModelsConfig(BaseModel):
    granite: ModelProfileConfig
    jina: ModelProfileConfig


class SamplingConfig(BaseModel):
    core_documents: int = 150
    granite_extended_documents: int = 50
    minimum_usable_questions: int = 1
    length_bins: int = 3


class DenseEncodingConfig(BaseModel):
    pooling: str = "mean"
    normalize: bool = True
    batch_size: int = 8
    device: str = "auto"
    dtype: str = "auto"
    attn_implementation: str = "sdpa"
    output_dtype: str = "float32"

    @model_validator(mode="after")
    def validate_dense_settings(self) -> DenseEncodingConfig:
        if self.pooling != "mean":
            raise ValueError("This controlled experiment fixes dense pooling at mean")
        if self.batch_size <= 0:
            raise ValueError("dense.batch_size must be positive")
        if self.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("dense.dtype must be auto, float32, float16, or bfloat16")
        if self.attn_implementation not in {
            "auto",
            "eager",
            "sdpa",
            "flash_attention_2",
            "flex_attention",
        }:
            raise ValueError(
                "dense.attn_implementation must be auto, eager, sdpa, "
                "flash_attention_2, or flex_attention"
            )
        if self.output_dtype not in {"float32", "float16"}:
            raise ValueError("dense.output_dtype must be float32 or float16")
        return self


class BM25Config(BaseModel):
    lowercase: bool = True
    token_pattern: str = r"(?u)\b\w[\w'-]*\b"
    k1: float = 1.5
    b: float = 0.75

    @model_validator(mode="after")
    def validate_parameters(self) -> BM25Config:
        if self.k1 <= 0:
            raise ValueError("BM25 k1 must be positive")
        if not 0 <= self.b <= 1:
            raise ValueError("BM25 b must be between 0 and 1")
        return self


class RankingConfig(BaseModel):
    max_depth: int = 10
    store_complete_ranking: bool = True

    @model_validator(mode="after")
    def validate_ranking(self) -> RankingConfig:
        if self.max_depth <= 0:
            raise ValueError("ranking.max_depth must be positive")
        return self


class EvaluationConfig(BaseModel):
    cutoffs: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    token_budgets: list[int] = Field(default_factory=lambda: [512, 1024, 2048, 4096])
    primary_metric: str = "ndcg_at_5"
    bootstrap_iterations: int = 10000
    confidence_level: float = 0.95
    bootstrap_metrics: list[str] = Field(
        default_factory=lambda: [
            "ndcg_at_5",
            "recall_at_5",
            "evidence_paragraph_recall_at_5",
            "complete_evidence_at_5",
        ]
    )
    query_types_filename: str = "query_types.csv"
    confirmatory_split: str = "test"
    require_query_types: bool = True
    error_analysis_per_direction: int = 10

    @model_validator(mode="after")
    def validate_evaluation(self) -> EvaluationConfig:
        if not self.cutoffs or any(cutoff <= 0 for cutoff in self.cutoffs):
            raise ValueError("evaluation.cutoffs must contain positive integers")
        if self.cutoffs != sorted(set(self.cutoffs)):
            raise ValueError("evaluation.cutoffs must be sorted and unique")
        if not self.token_budgets or any(budget <= 0 for budget in self.token_budgets):
            raise ValueError("evaluation.token_budgets must contain positive integers")
        if self.token_budgets != sorted(set(self.token_budgets)):
            raise ValueError("evaluation.token_budgets must be sorted and unique")
        if self.bootstrap_iterations <= 0:
            raise ValueError("evaluation.bootstrap_iterations must be positive")
        if not 0 < self.confidence_level < 1:
            raise ValueError("evaluation.confidence_level must be between 0 and 1")
        if 5 not in self.cutoffs or 10 not in self.cutoffs:
            raise ValueError("evaluation.cutoffs must include both 5 and 10")
        if not self.bootstrap_metrics:
            raise ValueError("evaluation.bootstrap_metrics cannot be empty")
        valid_metrics = {
            "average_precision",
            "reciprocal_rank",
            "r_precision",
            "full_ndcg",
            "normalized_first_relevant_rank",
            "first_relevant_rank_percentile",
        }
        for cutoff in self.cutoffs:
            for prefix in (
                "ndcg_at_",
                "precision_at_",
                "strict_precision_at_",
                "recall_at_",
                "evidence_paragraph_recall_at_",
                "evidence_span_coverage_at_",
                "complete_evidence_at_",
            ):
                valid_metrics.add(f"{prefix}{cutoff}")
        for budget in self.token_budgets:
            for prefix in (
                "evidence_paragraph_recall_at_token_budget_",
                "evidence_span_coverage_at_token_budget_",
                "complete_evidence_at_token_budget_",
            ):
                valid_metrics.add(f"{prefix}{budget}")
        alternative_evidence_metrics = {f"best_{metric}" for metric in valid_metrics}
        alternative_evidence_metrics.update(
            f"union_complete_evidence_at_{cutoff}" for cutoff in self.cutoffs
        )
        alternative_evidence_metrics.update(
            f"union_complete_evidence_at_token_budget_{budget}"
            for budget in self.token_budgets
        )
        valid_metrics.update(alternative_evidence_metrics)
        if self.primary_metric not in valid_metrics:
            raise ValueError(f"Unsupported evaluation.primary_metric: {self.primary_metric}")
        invalid_bootstrap = set(self.bootstrap_metrics).difference(valid_metrics)
        if invalid_bootstrap:
            raise ValueError(
                f"Unsupported evaluation.bootstrap_metrics: {sorted(invalid_bootstrap)}"
            )
        if self.primary_metric not in self.bootstrap_metrics:
            self.bootstrap_metrics = [*self.bootstrap_metrics, self.primary_metric]
        if not self.confirmatory_split:
            raise ValueError("evaluation.confirmatory_split cannot be empty")
        if self.error_analysis_per_direction <= 0:
            raise ValueError("evaluation.error_analysis_per_direction must be positive")
        return self



class AppConfig(BaseModel):
    project: ProjectConfig
    paths: PathConfig
    dataset: DatasetConfig
    document: DocumentConfig
    chunking: ChunkingConfig
    models: ModelsConfig
    sampling: SamplingConfig
    dense: DenseEncodingConfig = DenseEncodingConfig()
    bm25: BM25Config = BM25Config()
    ranking: RankingConfig = RankingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()

    @model_validator(mode="after")
    def validate_pipeline_depths(self) -> AppConfig:
        if (
            not self.ranking.store_complete_ranking
            and self.ranking.max_depth < max(self.evaluation.cutoffs)
        ):
            raise ValueError(
                "ranking.max_depth must be at least the largest evaluation cutoff"
            )

        return self


def load_config(path: Path) -> AppConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    return AppConfig.model_validate(raw)


def ensure_output_directories(config: AppConfig) -> None:
    for path in (
        config.paths.processed_dir,
        config.paths.profile_dir,
        config.paths.subset_dir,
        config.paths.retrieval_unit_dir,
        config.paths.encoding_dir,
        config.paths.ranking_dir,
        config.paths.evaluation_dir,
        config.paths.analysis_dir,
        config.paths.hf_cache_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
