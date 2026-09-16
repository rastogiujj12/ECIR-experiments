from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from sclc.data.schema import (
    DocumentRecord,
    PreparedQueryRecord,
    RelevanceJudgementRecord,
    RetrievalUnitRecord,
    TopLevelSectionRecord,
)


def write_models_jsonl(models: Iterable[BaseModel], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for model in models:
            handle.write(model.model_dump_json())
            handle.write("\n")


def read_retrieval_units(path: Path) -> Iterator[RetrievalUnitRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield RetrievalUnitRecord.model_validate(json.loads(line))



def read_prepared_queries(path: Path) -> Iterator[PreparedQueryRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield PreparedQueryRecord.model_validate(json.loads(line))


def read_top_level_sections(path: Path) -> Iterator[TopLevelSectionRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield TopLevelSectionRecord.model_validate(json.loads(line))

def relevance_frame(judgements: Iterable[RelevanceJudgementRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "query_id": judgement.query_id,
                "document_id": judgement.document_id,
                "segmentation_plan": judgement.segmentation_plan,
                "retrieval_unit_id": judgement.retrieval_unit_id,
                "relevance": judgement.relevance,
                "matched_evidence_paragraph_ids": "|".join(
                    judgement.matched_evidence_paragraph_ids
                ),
            }
            for judgement in judgements
        ]
    )



def query_type_coding_frame(
    documents: Iterable[DocumentRecord],
    queries: Iterable[PreparedQueryRecord],
) -> pd.DataFrame:
    documents_by_id = {document.document_id: document for document in documents}
    rows: list[dict[str, object]] = []
    for query in queries:
        document = documents_by_id[query.document_id]
        paragraphs = {paragraph.paragraph_id: paragraph for paragraph in document.paragraphs}
        sections = {section.section_id: section for section in document.sections}
        evidence_section_ids = sorted(
            {
                paragraphs[paragraph_id].section_id
                for paragraph_id in query.evidence_union_paragraph_ids
                if paragraph_id in paragraphs
            }
        )
        evidence_headings = sorted(
            {
                sections[section_id].top_level_heading or sections[section_id].heading
                for section_id in evidence_section_ids
                if section_id in sections
            }
        )
        rows.append(
            {
                "query_id": query.query_id,
                "document_id": query.document_id,
                "split": query.split,
                "analysis_set": query.analysis_set,
                "question": query.question,
                "evidence_set_count": len(query.evidence_sets),
                "evidence_paragraph_count": len(query.evidence_union_paragraph_ids),
                "evidence_section_count": len(evidence_section_ids),
                "evidence_section_headings": " | ".join(evidence_headings),
                "query_type": "",
                "coding_notes": "",
            }
        )
    return pd.DataFrame(rows)


def unit_summary_frame(units: Iterable[RetrievalUnitRecord]) -> pd.DataFrame:
    rows = []
    for unit in units:
        rows.append(
            {
                "retrieval_unit_id": unit.retrieval_unit_id,
                "document_id": unit.document_id,
                "analysis_set": unit.analysis_set,
                "segmentation_plan": unit.segmentation_plan,
                "unit_index": unit.unit_index,
                "character_start": unit.span.start,
                "character_end": unit.span.end,
                "character_count": unit.span.end - unit.span.start,
                "token_count": unit.token_count,
                "parent_section_id": unit.parent_section_id,
                "parent_section_heading": unit.parent_section_heading,
                "overlapping_section_count": len(unit.overlapping_section_ids),
                "overlapping_parent_section_count": len(
                    unit.overlapping_parent_section_ids
                ),
                "overlapping_paragraph_count": len(unit.overlapping_paragraph_ids),
                "fully_contained_paragraph_count": len(
                    unit.fully_contained_paragraph_ids
                ),
            }
        )
    return pd.DataFrame(rows)


def document_plan_summary_frame(units: Iterable[RetrievalUnitRecord]) -> pd.DataFrame:
    detail = unit_summary_frame(units)
    if detail.empty:
        return detail

    summary = (
        detail.groupby(
            ["document_id", "analysis_set", "segmentation_plan"],
            as_index=False,
        )
        .agg(
            unit_count=("retrieval_unit_id", "count"),
            minimum_tokens=("token_count", "min"),
            mean_tokens=("token_count", "mean"),
            maximum_tokens=("token_count", "max"),
            cross_section_unit_count=(
                "overlapping_parent_section_count",
                lambda values: int((values > 1).sum()),
            ),
            units_without_paragraphs=(
                "overlapping_paragraph_count",
                lambda values: int((values == 0).sum()),
            ),
        )
    )
    return summary


__all__ = [
    "PreparedQueryRecord",
    "RelevanceJudgementRecord",
    "RetrievalUnitRecord",
    "TopLevelSectionRecord",
    "document_plan_summary_frame",
    "read_prepared_queries",
    "read_retrieval_units",
    "query_type_coding_frame",
    "read_top_level_sections",
    "relevance_frame",
    "unit_summary_frame",
    "write_models_jsonl",
]
