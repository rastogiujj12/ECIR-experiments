from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

import pandas as pd
from rich.console import Console
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from sclc.config import AppConfig
from sclc.data.chunking import (
    ChunkingStats,
    build_continuous_units,
    build_section_bounded_units,
    build_top_level_sections,
    span_contains,
)
from sclc.data.io import read_documents_jsonl
from sclc.data.relevance import build_relevance_judgements, prepare_queries
from sclc.data.retrieval_unit_io import (
    document_plan_summary_frame,
    query_type_coding_frame,
    relevance_frame,
    unit_summary_frame,
    write_models_jsonl,
)
from sclc.data.schema import (
    DocumentRecord,
    PreparedQueryRecord,
    RelevanceJudgementRecord,
    RetrievalUnitRecord,
    TopLevelSectionRecord,
)
from sclc.paths import global_query_coding_path, retrieval_unit_dir

console = Console()


def load_canonical_tokenizer(config: AppConfig) -> PreTrainedTokenizerBase:
    tokenizer_kwargs: dict[str, Any] = {
        "trust_remote_code": config.chunking.tokenizer_trust_remote_code,
        "cache_dir": str(config.paths.hf_cache_dir),
        "use_fast": True,
    }
    if config.chunking.canonical_tokenizer_revision is not None:
        tokenizer_kwargs["revision"] = config.chunking.canonical_tokenizer_revision
    tokenizer = AutoTokenizer.from_pretrained(
        config.chunking.canonical_tokenizer,
        **tokenizer_kwargs,
    )
    if not tokenizer.is_fast:
        raise RuntimeError(
            f"{config.chunking.canonical_tokenizer} did not load a fast tokenizer. "
            "Fast character offsets are required for canonical chunk construction."
        )
    return tokenizer


def _load_selected_documents(
    config: AppConfig,
) -> tuple[list[DocumentRecord], dict[str, str]]:
    manifest_path = config.paths.subset_dir / "selected_documents.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} does not exist. Run `sclc sample` first."
        )

    manifest = pd.read_csv(manifest_path, dtype={"document_id": "string"})
    required = {"document_id", "analysis_set"}
    missing_columns = required.difference(manifest.columns)
    if missing_columns:
        raise ValueError(
            f"Subset manifest is missing columns: {sorted(missing_columns)}"
        )

    analysis_sets = {
        str(row.document_id): str(row.analysis_set)
        for row in manifest[["document_id", "analysis_set"]].itertuples(index=False)
    }
    selected_ids = set(analysis_sets)

    documents = [
        document
        for document in read_documents_jsonl(
            config.paths.processed_dir / "documents.jsonl"
        )
        if document.document_id in selected_ids
    ]
    found_ids = {document.document_id for document in documents}
    missing_ids = selected_ids.difference(found_ids)
    if missing_ids:
        preview = sorted(missing_ids)[:10]
        raise RuntimeError(
            f"Could not find {len(missing_ids)} selected papers in documents.jsonl. "
            f"First missing IDs: {preview}"
        )

    document_order = {
        document_id: index
        for index, document_id in enumerate(manifest["document_id"])
    }
    documents.sort(key=lambda document: document_order[document.document_id])
    return documents, analysis_sets


def _stats_to_dict(stats: ChunkingStats) -> dict[str, int]:
    return {
        "scope_count": stats.scope_count,
        "expected_tokens": stats.expected_tokens,
        "emitted_tokens": stats.emitted_tokens,
        "adjusted_boundaries": stats.adjusted_boundaries,
        "zero_length_offsets": stats.zero_length_offsets,
    }


def _validate(
    *,
    config: AppConfig,
    chunk_size_tokens: int,
    documents: list[DocumentRecord],
    parent_sections: list[TopLevelSectionRecord],
    continuous_units: list[RetrievalUnitRecord],
    section_units: list[RetrievalUnitRecord],
    queries: list[PreparedQueryRecord],
    judgements: list[RelevanceJudgementRecord],
    stats_by_document: dict[str, dict[str, ChunkingStats]],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    all_units = continuous_units + section_units
    documents_by_id = {document.document_id: document for document in documents}
    parents_by_id = {parent.parent_section_id: parent for parent in parent_sections}

    unit_ids = [unit.retrieval_unit_id for unit in all_units]
    duplicate_unit_ids = [
        unit_id for unit_id, count in Counter(unit_ids).items() if count > 1
    ]
    if duplicate_unit_ids:
        errors.append(f"Duplicate retrieval-unit IDs: {duplicate_unit_ids[:10]}")

    for unit in all_units:
        document = documents_by_id.get(unit.document_id)
        if document is None:
            errors.append(f"Unit {unit.retrieval_unit_id} has an unknown document")
            continue
        if not (0 <= unit.span.start < unit.span.end <= len(document.text)):
            errors.append(f"Invalid character span for {unit.retrieval_unit_id}")
            continue
        if document.text[unit.span.start : unit.span.end] != unit.text:
            errors.append(f"Stored text mismatch for {unit.retrieval_unit_id}")
        if not 1 <= unit.token_count <= chunk_size_tokens:
            errors.append(
                f"Token count {unit.token_count} is invalid for {unit.retrieval_unit_id}"
            )
        if unit.scope_token_end <= unit.scope_token_start:
            errors.append(f"Empty token interval for {unit.retrieval_unit_id}")

        if unit.segmentation_plan == "section_bounded":
            if unit.parent_section_id is None:
                errors.append(f"Missing parent section for {unit.retrieval_unit_id}")
                continue
            parent = parents_by_id.get(unit.parent_section_id)
            if parent is None:
                errors.append(f"Unknown parent section for {unit.retrieval_unit_id}")
                continue
            if not span_contains(parent.span, unit.span):
                errors.append(f"Section boundary crossed by {unit.retrieval_unit_id}")
            if unit.overlapping_parent_section_ids != [unit.parent_section_id]:
                errors.append(
                    f"Unexpected top-level section overlap for {unit.retrieval_unit_id}: "
                    f"{unit.overlapping_parent_section_ids}"
                )

    for document in documents:
        for plan in ("continuous", "section_bounded"):
            plan_stats = stats_by_document[document.document_id][plan]
            if config.chunking.overlap_tokens == 0 and (
                plan_stats.expected_tokens != plan_stats.emitted_tokens
            ):
                errors.append(
                    f"Canonical token coverage mismatch for {document.document_id}/{plan}: "
                    f"expected {plan_stats.expected_tokens}, emitted {plan_stats.emitted_tokens}"
                )
            if plan_stats.zero_length_offsets:
                errors.append(
                    f"Zero-length offsets found for {document.document_id}/{plan}"
                )

    query_ids = [query.query_id for query in queries]
    duplicate_query_ids = [
        query_id for query_id, count in Counter(query_ids).items() if count > 1
    ]
    if duplicate_query_ids:
        errors.append(
            f"Duplicate query IDs in selected corpus: {duplicate_query_ids[:10]}"
        )

    units_by_plan = {
        "continuous": continuous_units,
        "section_bounded": section_units,
    }
    unit_id_set = set(unit_ids)
    relevant_by_query_plan: defaultdict[tuple[str, str], int] = defaultdict(int)
    for judgement in judgements:
        if judgement.retrieval_unit_id not in unit_id_set:
            errors.append(
                f"Qrel refers to unknown unit {judgement.retrieval_unit_id}"
            )
        relevant_by_query_plan[(judgement.query_id, judgement.segmentation_plan)] += 1

    for query in queries:
        document = documents_by_id[query.document_id]
        paragraph_ids = {paragraph.paragraph_id for paragraph in document.paragraphs}
        missing_evidence = set(query.evidence_union_paragraph_ids).difference(paragraph_ids)
        if missing_evidence:
            errors.append(
                f"Query {query.query_id} refers to unknown evidence paragraphs: "
                f"{sorted(missing_evidence)}"
            )
        for plan in units_by_plan:
            if relevant_by_query_plan[(query.query_id, plan)] == 0:
                errors.append(
                    f"Query {query.query_id} has no relevant units under {plan}"
                )

    cross_parent_count = sum(
        len(unit.overlapping_parent_section_ids) > 1 for unit in continuous_units
    )
    if cross_parent_count == 0:
        warnings.append(
            "No continuous unit crossed a top-level section boundary in the selected sample."
        )

    units_without_paragraphs = sum(
        not unit.overlapping_paragraph_ids for unit in all_units
    )
    if units_without_paragraphs:
        warnings.append(
            f"{units_without_paragraphs} retrieval units contain headings or other text "
            "but do not overlap a QASPER paragraph."
        )

    return {
        "status": "passed" if not errors else "failed",
        "configuration": {
            "canonical_tokenizer": config.chunking.canonical_tokenizer,
            "chunk_size_tokens": chunk_size_tokens,
            "overlap_tokens": config.chunking.overlap_tokens,
            "retain_short_final_chunk": config.chunking.retain_short_final_chunk,
        },
        "counts": {
            "documents": len(documents),
            "top_level_sections": len(parent_sections),
            "continuous_units": len(continuous_units),
            "section_bounded_units": len(section_units),
            "usable_queries": len(queries),
            "relevance_judgements": len(judgements),
            "continuous_units_crossing_top_level_boundaries": cross_parent_count,
            "units_without_paragraph_overlap": units_without_paragraphs,
        },
        "errors": errors,
        "warnings": warnings,
        "per_document_token_coverage": {
            document_id: {
                plan: _stats_to_dict(stats)
                for plan, stats in plan_stats.items()
            }
            for document_id, plan_stats in stats_by_document.items()
        },
    }


def construct_retrieval_units(
    config: AppConfig,
    *,
    chunk_size_tokens: int,
) -> dict[str, Any]:
    documents, analysis_sets = _load_selected_documents(config)
    tokenizer = load_canonical_tokenizer(config)

    parent_sections: list[TopLevelSectionRecord] = []
    continuous_units: list[RetrievalUnitRecord] = []
    section_units: list[RetrievalUnitRecord] = []
    queries: list[PreparedQueryRecord] = []
    stats_by_document: dict[str, dict[str, ChunkingStats]] = {}

    for document in tqdm(documents, desc="Constructing retrieval units"):
        analysis_set = analysis_sets[document.document_id]
        parents = build_top_level_sections(document, analysis_set)
        parent_sections.extend(parents)

        document_continuous, continuous_stats = build_continuous_units(
            document=document,
            analysis_set=analysis_set,
            tokenizer=tokenizer,
            chunk_size_tokens=chunk_size_tokens,
            overlap_tokens=config.chunking.overlap_tokens,
        )
        document_section, section_stats = build_section_bounded_units(
            document=document,
            analysis_set=analysis_set,
            tokenizer=tokenizer,
            chunk_size_tokens=chunk_size_tokens,
            overlap_tokens=config.chunking.overlap_tokens,
            parent_sections=parents,
        )
        continuous_units.extend(document_continuous)
        section_units.extend(document_section)
        queries.extend(prepare_queries(document, analysis_set))
        stats_by_document[document.document_id] = {
            "continuous": continuous_stats,
            "section_bounded": section_stats,
        }

    continuous_qrels = build_relevance_judgements(queries, continuous_units)
    section_qrels = build_relevance_judgements(queries, section_units)
    judgements = continuous_qrels + section_qrels

    output_dir = retrieval_unit_dir(config, chunk_size_tokens)
    write_models_jsonl(parent_sections, output_dir / "top_level_sections.jsonl")
    write_models_jsonl(continuous_units, output_dir / "continuous_units.jsonl")
    write_models_jsonl(section_units, output_dir / "section_bounded_units.jsonl")
    write_models_jsonl(queries, output_dir / "queries.jsonl")
    coding_frame = query_type_coding_frame(documents, queries)
    coding_frame.to_csv(output_dir / "query_type_coding.csv", index=False)
    global_coding_path = global_query_coding_path(config)
    if not global_coding_path.exists():
        global_coding_path.parent.mkdir(parents=True, exist_ok=True)
        coding_frame.to_csv(global_coding_path, index=False)

    relevance = relevance_frame(judgements)
    if relevance.empty:
        relevance = pd.DataFrame(
            columns=[
                "query_id",
                "document_id",
                "segmentation_plan",
                "retrieval_unit_id",
                "relevance",
                "matched_evidence_paragraph_ids",
            ]
        )
    relevance.to_csv(output_dir / "qrels.csv", index=False)

    all_units = continuous_units + section_units
    unit_summary_frame(all_units).to_csv(
        output_dir / "retrieval_units.csv", index=False
    )
    document_plan_summary_frame(all_units).to_csv(
        output_dir / "retrieval_unit_summary.csv", index=False
    )

    validation = _validate(
        config=config,
        chunk_size_tokens=chunk_size_tokens,
        documents=documents,
        parent_sections=parent_sections,
        continuous_units=continuous_units,
        section_units=section_units,
        queries=queries,
        judgements=judgements,
        stats_by_document=stats_by_document,
    )
    with (output_dir / "chunking_validation.json").open("w", encoding="utf-8") as handle:
        json.dump(validation, handle, indent=2)

    if validation["status"] != "passed":
        preview = validation["errors"][:10]
        raise RuntimeError(
            "Retrieval-unit validation failed. See chunking_validation.json. "
            f"First errors: {preview}"
        )

    return validation
