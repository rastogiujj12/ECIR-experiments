from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd

from sclc.config import AppConfig
from sclc.data.io import read_documents_jsonl
from sclc.data.query_types import load_query_types
from sclc.data.retrieval_unit_io import read_prepared_queries, read_retrieval_units
from sclc.data.schema import (
    DocumentRecord,
    PreparedQueryRecord,
    RankingRecord,
    RetrievalUnitRecord,
)
from sclc.options import EmbeddingModel, RetrievalCondition
from sclc.paths import evaluation_dir, ranking_dir, retrieval_unit_dir
from sclc.retrieval.ranking import read_rankings


def _fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _ranking_dir(
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


def _evaluation_dir(
    config: AppConfig,
    condition: RetrievalCondition,
    model: EmbeddingModel | None,
    *,
    chunk_size: int,
) -> Path:
    base = evaluation_dir(config, chunk_size)
    if condition is RetrievalCondition.BM25:
        return base / condition.value
    assert model is not None
    return base / condition.value / model.value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run the preceding pipeline stage first."
        )
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _eligible_analysis_set(model: EmbeddingModel | None, analysis_set: str) -> bool:
    return model is None or model is EmbeddingModel.GRANITE or analysis_set == "cross_model_core"


def _load_queries(
    config: AppConfig,
    model: EmbeddingModel | None,
    *,
    chunk_size: int,
) -> list[PreparedQueryRecord]:
    path = retrieval_unit_dir(config, chunk_size) / "queries.jsonl"
    return [
        query
        for query in read_prepared_queries(path)
        if _eligible_analysis_set(model, query.analysis_set)
    ]


def _load_units(
    config: AppConfig,
    condition: RetrievalCondition,
    model: EmbeddingModel | None,
    *,
    chunk_size: int,
) -> list[RetrievalUnitRecord]:
    filename = (
        "continuous_units.jsonl"
        if condition in {RetrievalCondition.BM25, RetrievalCondition.FIXED_DENSE}
        else "section_bounded_units.jsonl"
    )
    path = retrieval_unit_dir(config, chunk_size) / filename
    return [
        unit
        for unit in read_retrieval_units(path)
        if _eligible_analysis_set(model, unit.analysis_set)
    ]


def _load_length_strata(config: AppConfig) -> dict[str, str]:
    path = config.paths.subset_dir / "selected_documents.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path, dtype={"document_id": "string"})
    if "length_stratum" not in frame.columns:
        return {}
    return {
        str(row.document_id): str(row.length_stratum)
        for row in frame[["document_id", "length_stratum"]].itertuples(index=False)
    }


def _dcg(relevance: Sequence[int]) -> float:
    return sum(value / math.log2(index + 2) for index, value in enumerate(relevance))


def ndcg_at_k(
    ranked_unit_ids: Sequence[str],
    relevant_unit_ids: set[str],
    k: int,
) -> float:
    if not relevant_unit_ids or k <= 0:
        return 0.0
    observed = [
        1 if unit_id in relevant_unit_ids else 0 for unit_id in ranked_unit_ids[:k]
    ]
    ideal = [1] * min(len(relevant_unit_ids), k)
    denominator = _dcg(ideal)
    return _dcg(observed) / denominator if denominator else 0.0


def precision_at_k(
    ranked_unit_ids: Sequence[str],
    relevant_unit_ids: set[str],
    k: int,
) -> float:
    """Precision over the available prefix, equivalent to P@min(k, N)."""
    top = ranked_unit_ids[:k]
    if not top:
        return 0.0
    return sum(unit_id in relevant_unit_ids for unit_id in top) / len(top)


def strict_precision_at_k(
    ranked_unit_ids: Sequence[str],
    relevant_unit_ids: set[str],
    k: int,
) -> float:
    """Precision with k as the denominator even when the paper has fewer candidates."""
    if k <= 0:
        return 0.0
    return sum(unit_id in relevant_unit_ids for unit_id in ranked_unit_ids[:k]) / k


def recall_at_k(
    ranked_unit_ids: Sequence[str],
    relevant_unit_ids: set[str],
    k: int,
) -> float:
    if not relevant_unit_ids:
        return 0.0
    return sum(
        unit_id in relevant_unit_ids for unit_id in ranked_unit_ids[:k]
    ) / len(relevant_unit_ids)


def average_precision(
    ranked_unit_ids: Sequence[str],
    relevant_unit_ids: set[str],
) -> float:
    if not relevant_unit_ids:
        return 0.0
    hits = 0
    total = 0.0
    for rank, unit_id in enumerate(ranked_unit_ids, start=1):
        if unit_id in relevant_unit_ids:
            hits += 1
            total += hits / rank
    return total / len(relevant_unit_ids)


def reciprocal_rank(
    ranked_unit_ids: Sequence[str],
    relevant_unit_ids: set[str],
) -> float:
    for rank, unit_id in enumerate(ranked_unit_ids, start=1):
        if unit_id in relevant_unit_ids:
            return 1.0 / rank
    return 0.0


def r_precision(
    ranked_unit_ids: Sequence[str],
    relevant_unit_ids: set[str],
) -> float:
    relevant_count = len(relevant_unit_ids)
    if relevant_count == 0:
        return 0.0
    return sum(
        unit_id in relevant_unit_ids for unit_id in ranked_unit_ids[:relevant_count]
    ) / relevant_count


def first_relevant_rank(
    ranked_unit_ids: Sequence[str],
    relevant_unit_ids: set[str],
) -> int:
    for rank, unit_id in enumerate(ranked_unit_ids, start=1):
        if unit_id in relevant_unit_ids:
            return rank
    return len(ranked_unit_ids) + 1


def _covered_paragraphs(
    ranked_unit_ids: Sequence[str],
    units_by_id: dict[str, RetrievalUnitRecord],
    evidence_paragraph_ids: set[str],
    k: int,
) -> set[str]:
    covered: set[str] = set()
    for unit_id in ranked_unit_ids[:k]:
        covered.update(
            evidence_paragraph_ids.intersection(
                units_by_id[unit_id].overlapping_paragraph_ids
            )
        )
    return covered


def evidence_paragraph_recall_at_k(
    ranked_unit_ids: Sequence[str],
    units_by_id: dict[str, RetrievalUnitRecord],
    evidence_paragraph_ids: set[str],
    k: int,
) -> float:
    if not evidence_paragraph_ids:
        return 0.0
    covered = _covered_paragraphs(
        ranked_unit_ids, units_by_id, evidence_paragraph_ids, k
    )
    return len(covered) / len(evidence_paragraph_ids)


def complete_evidence_at_k(
    ranked_unit_ids: Sequence[str],
    units_by_id: dict[str, RetrievalUnitRecord],
    evidence_paragraph_ids: set[str],
    k: int,
) -> float:
    if not evidence_paragraph_ids:
        return 0.0
    return float(
        _covered_paragraphs(ranked_unit_ids, units_by_id, evidence_paragraph_ids, k)
        == evidence_paragraph_ids
    )


def _non_whitespace_positions(text: str, start: int, end: int) -> set[int]:
    return {
        index
        for index in range(start, end)
        if index < len(text) and not text[index].isspace()
    }


def evidence_span_coverage_at_k(
    ranked_unit_ids: Sequence[str],
    units_by_id: dict[str, RetrievalUnitRecord],
    paragraph_spans: dict[str, tuple[int, int]],
    evidence_paragraph_ids: set[str],
    k: int,
    document_text: str,
) -> float:
    evidence_positions: set[int] = set()
    for paragraph_id in evidence_paragraph_ids:
        start, end = paragraph_spans[paragraph_id]
        evidence_positions.update(_non_whitespace_positions(document_text, start, end))
    if not evidence_positions:
        return 0.0

    covered_positions: set[int] = set()
    for unit_id in ranked_unit_ids[:k]:
        unit = units_by_id[unit_id]
        covered_positions.update(
            position
            for position in evidence_positions
            if unit.span.start <= position < unit.span.end
        )
    return len(covered_positions) / len(evidence_positions)


def _prefix_for_token_budget(
    ranked_unit_ids: Sequence[str],
    units_by_id: dict[str, RetrievalUnitRecord],
    budget: int,
) -> tuple[list[str], int]:
    selected: list[str] = []
    tokens = 0
    for unit_id in ranked_unit_ids:
        unit_tokens = units_by_id[unit_id].token_count
        if tokens + unit_tokens > budget:
            break
        selected.append(unit_id)
        tokens += unit_tokens
    return selected, tokens


def _relevant_units_for_paragraphs(
    units: Sequence[RetrievalUnitRecord],
    paragraph_ids: set[str],
) -> set[str]:
    return {
        unit.retrieval_unit_id
        for unit in units
        if paragraph_ids.intersection(unit.overlapping_paragraph_ids)
    }


def _score_evidence_set(
    *,
    ranked_unit_ids: Sequence[str],
    units: Sequence[RetrievalUnitRecord],
    units_by_id: dict[str, RetrievalUnitRecord],
    paragraph_spans: dict[str, tuple[int, int]],
    document_text: str,
    evidence_paragraph_ids: set[str],
    cutoffs: Sequence[int],
    token_budgets: Sequence[int],
) -> dict[str, float]:
    relevant_unit_ids = _relevant_units_for_paragraphs(units, evidence_paragraph_ids)
    scores: dict[str, float] = {}
    for cutoff in cutoffs:
        scores[f"ndcg_at_{cutoff}"] = ndcg_at_k(
            ranked_unit_ids, relevant_unit_ids, cutoff
        )
        scores[f"precision_at_{cutoff}"] = precision_at_k(
            ranked_unit_ids, relevant_unit_ids, cutoff
        )
        scores[f"strict_precision_at_{cutoff}"] = strict_precision_at_k(
            ranked_unit_ids, relevant_unit_ids, cutoff
        )
        scores[f"recall_at_{cutoff}"] = recall_at_k(
            ranked_unit_ids, relevant_unit_ids, cutoff
        )
        scores[f"evidence_paragraph_recall_at_{cutoff}"] = (
            evidence_paragraph_recall_at_k(
                ranked_unit_ids, units_by_id, evidence_paragraph_ids, cutoff
            )
        )
        scores[f"evidence_span_coverage_at_{cutoff}"] = evidence_span_coverage_at_k(
            ranked_unit_ids,
            units_by_id,
            paragraph_spans,
            evidence_paragraph_ids,
            cutoff,
            document_text,
        )
        scores[f"complete_evidence_at_{cutoff}"] = complete_evidence_at_k(
            ranked_unit_ids, units_by_id, evidence_paragraph_ids, cutoff
        )

    scores["average_precision"] = average_precision(
        ranked_unit_ids, relevant_unit_ids
    )
    scores["reciprocal_rank"] = reciprocal_rank(
        ranked_unit_ids, relevant_unit_ids
    )
    scores["r_precision"] = r_precision(ranked_unit_ids, relevant_unit_ids)
    scores["full_ndcg"] = ndcg_at_k(
        ranked_unit_ids, relevant_unit_ids, len(ranked_unit_ids)
    )
    first_rank = first_relevant_rank(ranked_unit_ids, relevant_unit_ids)
    candidate_count = len(ranked_unit_ids)
    scores["first_relevant_rank"] = float(first_rank)
    scores["normalized_first_relevant_rank"] = (
        (first_rank - 1) / max(candidate_count - 1, 1)
    )
    scores["first_relevant_rank_percentile"] = (
        first_rank / candidate_count if candidate_count else 1.0
    )

    for budget in token_budgets:
        prefix, _ = _prefix_for_token_budget(ranked_unit_ids, units_by_id, budget)
        prefix_count = len(prefix)
        scores[f"evidence_paragraph_recall_at_token_budget_{budget}"] = (
            evidence_paragraph_recall_at_k(
                prefix, units_by_id, evidence_paragraph_ids, prefix_count
            )
        )
        scores[f"evidence_span_coverage_at_token_budget_{budget}"] = (
            evidence_span_coverage_at_k(
                prefix,
                units_by_id,
                paragraph_spans,
                evidence_paragraph_ids,
                prefix_count,
                document_text,
            )
        )
        scores[f"complete_evidence_at_token_budget_{budget}"] = complete_evidence_at_k(
            prefix, units_by_id, evidence_paragraph_ids, prefix_count
        )
    return scores


def _best_scores(score_sets: Sequence[dict[str, float]]) -> dict[str, float]:
    if not score_sets:
        raise ValueError("A retained query must contain at least one evidence set")
    lower_is_better = {
        "first_relevant_rank",
        "normalized_first_relevant_rank",
        "first_relevant_rank_percentile",
    }
    return {
        key: (
            min(scores[key] for scores in score_sets)
            if key in lower_is_better
            else max(scores[key] for scores in score_sets)
        )
        for key in score_sets[0]
    }


def _paragraph_spans_by_document(
    documents: Iterable[DocumentRecord],
) -> dict[str, dict[str, tuple[int, int]]]:
    return {
        document.document_id: {
            paragraph.paragraph_id: (paragraph.span.start, paragraph.span.end)
            for paragraph in document.paragraphs
        }
        for document in documents
    }


def _document_text_hashes(documents: Iterable[DocumentRecord]) -> dict[str, str]:
    return {
        document.document_id: hashlib.sha256(
            document.text.encode("utf-8")
        ).hexdigest()
        for document in documents
    }


def _evaluation_inputs_fingerprint(
    *,
    queries: Sequence[PreparedQueryRecord],
    units: Sequence[RetrievalUnitRecord],
    rankings: Sequence[RankingRecord],
    paragraph_spans_by_document: dict[str, dict[str, tuple[int, int]]],
    document_text_hashes: dict[str, str],
) -> str:
    payload = {
        "queries": [query.model_dump(mode="json") for query in queries],
        "units": [
            {
                "retrieval_unit_id": unit.retrieval_unit_id,
                "document_id": unit.document_id,
                "span": [unit.span.start, unit.span.end],
                "token_count": unit.token_count,
                "overlapping_paragraph_ids": unit.overlapping_paragraph_ids,
            }
            for unit in units
        ],
        "rankings": [
            {
                "query_id": ranking.query_id,
                "retrieval_unit_id": ranking.retrieval_unit_id,
                "rank": ranking.rank,
                "score": ranking.score,
            }
            for ranking in rankings
        ],
        "paragraph_spans": paragraph_spans_by_document,
        "document_text_hashes": document_text_hashes,
    }
    return _fingerprint(payload)


def _summary_frame(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    exact_metrics = {
        "average_precision",
        "reciprocal_rank",
        "r_precision",
        "full_ndcg",
        "first_relevant_rank",
        "normalized_first_relevant_rank",
        "first_relevant_rank_percentile",
    }
    metric_columns = [
        column
        for column in frame.columns
        if column in exact_metrics
        or column.startswith(
            (
                "ndcg_",
                "precision_",
                "strict_precision_",
                "recall_",
                "evidence_",
                "complete_",
                "union_",
                "best_",
                "retrieved_fraction_",
                "retrieved_tokens_",
                "retrieved_chunks_",
                "cutoff_saturated_",
            )
        )
    ]
    grouped = frame.groupby(group_columns, dropna=False)[metric_columns].agg(
        ["mean", "std"]
    )
    grouped.columns = [
        f"{metric}_{statistic}" for metric, statistic in grouped.columns
    ]
    # A wide multi-metric aggregation can leave pandas with many internal
    # blocks.  reset_index() inserts the grouping columns and warns when the
    # frame is fragmented, so consolidate the blocks before resetting it.
    result = grouped.copy().reset_index()
    counts = frame.groupby(group_columns, dropna=False).agg(
        query_count=("query_id", "count"),
        document_count=("document_id", "nunique"),
    )
    return result.merge(counts.reset_index(), on=group_columns, how="left")


def evaluate_condition(
    config: AppConfig,
    *,
    condition: RetrievalCondition,
    model: EmbeddingModel | None,
    chunk_size: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    if condition is RetrievalCondition.BM25 and model is not None:
        raise ValueError("--model must not be supplied for BM25 evaluation")
    if condition.is_dense and model is None:
        raise ValueError(f"--model is required for {condition.value} evaluation")

    source_ranking_dir = _ranking_dir(
        config, condition, model, chunk_size=chunk_size
    )
    ranking_manifest = _read_json(source_ranking_dir / "manifest.json")
    rankings_path = source_ranking_dir / str(ranking_manifest["file"])
    output_dir = _evaluation_dir(
        config, condition, model, chunk_size=chunk_size
    )
    metrics_path = output_dir / "query_metrics.csv"
    manifest_path = output_dir / "manifest.json"

    queries = _load_queries(config, model, chunk_size=chunk_size)
    units = _load_units(config, condition, model, chunk_size=chunk_size)
    units_by_id = {unit.retrieval_unit_id: unit for unit in units}
    units_by_document: dict[str, list[RetrievalUnitRecord]] = defaultdict(list)
    for unit in units:
        units_by_document[unit.document_id].append(unit)
    rankings = read_rankings(rankings_path)
    rankings_by_query: dict[str, list[RankingRecord]] = defaultdict(list)
    for ranking in rankings:
        rankings_by_query[ranking.query_id].append(ranking)
    for records in rankings_by_query.values():
        records.sort(key=lambda item: item.rank)
        ranks = [item.rank for item in records]
        if ranks != list(range(1, len(ranks) + 1)):
            raise RuntimeError("Ranking positions must be consecutive and start at one")
        if len({item.retrieval_unit_id for item in records}) != len(records):
            raise RuntimeError("A ranking contains duplicate retrieval units")

    documents = list(
        read_documents_jsonl(config.paths.processed_dir / "documents.jsonl")
    )
    documents_by_id = {document.document_id: document for document in documents}
    paragraph_spans_by_document = _paragraph_spans_by_document(documents)
    document_text_hashes = _document_text_hashes(documents)
    query_types = load_query_types(config, {query.query_id for query in queries})
    length_strata = _load_length_strata(config)

    configuration = {
        "condition": condition.value,
        "chunk_size_tokens": chunk_size,
        "model_key": model.value if model is not None else None,
        "cutoffs": config.evaluation.cutoffs,
        "token_budgets": config.evaluation.token_budgets,
        "ranking_fingerprint": ranking_manifest["configuration_fingerprint"],
        "evaluation_inputs_fingerprint": _evaluation_inputs_fingerprint(
            queries=queries,
            units=units,
            rankings=rankings,
            paragraph_spans_by_document=paragraph_spans_by_document,
            document_text_hashes=document_text_hashes,
        ),
        "query_types": query_types,
    }
    fingerprint = _fingerprint(configuration)
    if metrics_path.exists() and manifest_path.exists() and not overwrite:
        existing = _read_json(manifest_path)
        if existing.get("configuration_fingerprint") != fingerprint:
            raise RuntimeError(
                f"Cached evaluation at {metrics_path} does not match current inputs. "
                "Re-run with --overwrite."
            )
        return existing

    rows: list[dict[str, Any]] = []
    for query in queries:
        query_rankings = rankings_by_query.get(query.query_id)
        if not query_rankings:
            raise RuntimeError(f"No ranking exists for query {query.query_id}")
        if any(item.document_id != query.document_id for item in query_rankings):
            raise RuntimeError(f"Query {query.query_id} retrieved from another paper")
        ranked_unit_ids = [item.retrieval_unit_id for item in query_rankings]
        if any(unit_id not in units_by_id for unit_id in ranked_unit_ids):
            raise RuntimeError(
                f"Query {query.query_id} ranking refers to an unknown unit"
            )
        document_units = units_by_document[query.document_id]
        if config.ranking.store_complete_ranking and len(ranked_unit_ids) != len(
            document_units
        ):
            raise RuntimeError(
                f"Query {query.query_id} has {len(ranked_unit_ids)} ranked units but "
                f"{len(document_units)} candidates. Re-run retrieval with complete rankings."
            )
        paragraph_spans = paragraph_spans_by_document[query.document_id]
        document_text = documents_by_id[query.document_id].text

        set_scores = [
            _score_evidence_set(
                ranked_unit_ids=ranked_unit_ids,
                units=document_units,
                units_by_id=units_by_id,
                paragraph_spans=paragraph_spans,
                document_text=document_text,
                evidence_paragraph_ids=set(evidence_set.paragraph_ids),
                cutoffs=config.evaluation.cutoffs,
                token_budgets=config.evaluation.token_budgets,
            )
            for evidence_set in query.evidence_sets
        ]
        best = _best_scores(set_scores)
        primary = _score_evidence_set(
            ranked_unit_ids=ranked_unit_ids,
            units=document_units,
            units_by_id=units_by_id,
            paragraph_spans=paragraph_spans,
            document_text=document_text,
            evidence_paragraph_ids=set(query.evidence_union_paragraph_ids),
            cutoffs=config.evaluation.cutoffs,
            token_budgets=config.evaluation.token_budgets,
        )
        for cutoff in config.evaluation.cutoffs:
            key = f"complete_evidence_at_{cutoff}"
            primary[key] = best[key]
        for budget in config.evaluation.token_budgets:
            key = f"complete_evidence_at_token_budget_{budget}"
            primary[key] = best[key]

        diagnostics: dict[str, Any] = {}
        candidate_count = len(document_units)
        for cutoff in config.evaluation.cutoffs:
            returned = min(cutoff, len(ranked_unit_ids))
            diagnostics[f"cutoff_saturated_at_{cutoff}"] = float(
                candidate_count <= cutoff
            )
            diagnostics[f"retrieved_fraction_at_{cutoff}"] = (
                returned / candidate_count if candidate_count else 0.0
            )
        for budget in config.evaluation.token_budgets:
            prefix, tokens = _prefix_for_token_budget(
                ranked_unit_ids, units_by_id, budget
            )
            diagnostics[f"retrieved_chunks_at_token_budget_{budget}"] = len(prefix)
            diagnostics[f"retrieved_tokens_at_token_budget_{budget}"] = tokens

        union_scores = _score_evidence_set(
            ranked_unit_ids=ranked_unit_ids,
            units=document_units,
            units_by_id=units_by_id,
            paragraph_spans=paragraph_spans,
            document_text=document_text,
            evidence_paragraph_ids=set(query.evidence_union_paragraph_ids),
            cutoffs=config.evaluation.cutoffs,
            token_budgets=config.evaluation.token_budgets,
        )
        union_complete = {
            f"union_complete_evidence_at_{cutoff}": union_scores[
                f"complete_evidence_at_{cutoff}"
            ]
            for cutoff in config.evaluation.cutoffs
        }
        union_complete.update(
            {
                f"union_complete_evidence_at_token_budget_{budget}": union_scores[
                    f"complete_evidence_at_token_budget_{budget}"
                ]
                for budget in config.evaluation.token_budgets
            }
        )

        rows.append(
            {
                "query_id": query.query_id,
                "document_id": query.document_id,
                "question": query.question,
                "split": query.split,
                "analysis_set": query.analysis_set,
                "length_stratum": length_strata.get(
                    query.document_id, "unassigned"
                ),
                "query_type": query_types.get(
                    query.query_id, query.query_type or "unclassified"
                ),
                "condition": condition.value,
                "model_key": model.value if model is not None else "none",
                "chunk_size_tokens": chunk_size,
                "candidate_count": candidate_count,
                "returned_count": len(ranked_unit_ids),
                "evidence_set_count": len(query.evidence_sets),
                **diagnostics,
                **primary,
                **{f"best_{key}": value for key, value in best.items()},
                **union_complete,
            }
        )

    frame = pd.DataFrame(rows).sort_values(
        ["analysis_set", "document_id", "query_id"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(metrics_path, index=False)
    summary_overall = _summary_frame(frame, ["analysis_set"])
    category_frame = frame[
        ~frame["query_type"].isin(["uncertain", "unclassified"])
    ]
    summary_by_query_type = _summary_frame(
        category_frame, ["analysis_set", "query_type"]
    )
    summary_by_length = _summary_frame(
        frame, ["analysis_set", "length_stratum"]
    )
    summary_by_split = _summary_frame(frame, ["split", "analysis_set"])
    summary_overall.to_csv(output_dir / "summary_overall.csv", index=False)
    summary_by_query_type.to_csv(
        output_dir / "summary_by_query_type.csv", index=False
    )
    summary_by_length.to_csv(
        output_dir / "summary_by_length_stratum.csv", index=False
    )
    summary_by_split.to_csv(output_dir / "summary_by_split.csv", index=False)

    manifest = {
        "schema_version": 2,
        "kind": "retrieval_evaluation",
        "condition": condition.value,
        "model_key": model.value if model is not None else None,
        "chunk_size_tokens": chunk_size,
        "query_count": len(frame),
        "document_count": int(frame["document_id"].nunique()),
        "classified_query_count": int(
            (frame["query_type"] != "unclassified").sum()
        ),
        "cutoffs": config.evaluation.cutoffs,
        "token_budgets": config.evaluation.token_budgets,
        "primary_metric": config.evaluation.primary_metric,
        "evidence_policy": {
            "ranking_and_coverage": "union_of_distinct_evidence_paragraphs",
            "complete_support": "all_paragraphs_in_any_acceptable_evidence_set",
            "alternative_evidence_prefix": "best_",
        },
        "precision_policy": {
            "precision_at_k": "divide_by_available_prefix_min_k_candidate_count",
            "strict_precision_at_k": "divide_by_requested_k",
        },
        "configuration": configuration,
        "configuration_fingerprint": fingerprint,
        "files": {
            "query_metrics": metrics_path.name,
            "summary_overall": "summary_overall.csv",
            "summary_by_query_type": "summary_by_query_type.csv",
            "summary_by_length_stratum": "summary_by_length_stratum.csv",
            "summary_by_split": "summary_by_split.csv",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


__all__ = [
    "average_precision",
    "complete_evidence_at_k",
    "evaluate_condition",
    "evidence_paragraph_recall_at_k",
    "evidence_span_coverage_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "r_precision",
    "recall_at_k",
    "reciprocal_rank",
    "strict_precision_at_k",
]
