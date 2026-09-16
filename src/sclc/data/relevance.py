from __future__ import annotations

from collections.abc import Iterable, Sequence

from sclc.data.schema import (
    DocumentRecord,
    EvidenceSetRecord,
    PreparedQueryRecord,
    RelevanceJudgementRecord,
    RetrievalUnitRecord,
)


def _deduplicate_evidence_sets(evidence_sets: list[list[str]]) -> list[list[str]]:
    seen: set[tuple[str, ...]] = set()
    unique: list[list[str]] = []
    for paragraph_ids in evidence_sets:
        key = tuple(sorted(set(paragraph_ids)))
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(list(key))
    return unique


def prepare_queries(document: DocumentRecord, analysis_set: str) -> list[PreparedQueryRecord]:
    prepared: list[PreparedQueryRecord] = []

    for query in document.queries:
        evidence_sets: list[list[str]] = []
        for answer in query.answers:
            if answer.unanswerable:
                continue
            paragraph_ids = sorted(
                {
                    paragraph_id
                    for item in answer.evidence
                    if not item.is_float
                    for paragraph_id in item.matched_paragraph_ids
                }
            )
            if paragraph_ids:
                evidence_sets.append(paragraph_ids)

        unique_sets = _deduplicate_evidence_sets(evidence_sets)
        if not unique_sets:
            continue

        evidence_union = sorted(
            {paragraph_id for evidence_set in unique_sets for paragraph_id in evidence_set}
        )
        prepared.append(
            PreparedQueryRecord(
                query_id=query.query_id,
                document_id=document.document_id,
                split=document.split,
                analysis_set=analysis_set,
                question=query.question,
                evidence_union_paragraph_ids=evidence_union,
                evidence_sets=[
                    EvidenceSetRecord(
                        evidence_set_id=f"{query.query_id}:evidence_set:{index:02d}",
                        paragraph_ids=paragraph_ids,
                    )
                    for index, paragraph_ids in enumerate(unique_sets)
                ],
            )
        )

    return prepared


def build_relevance_judgements(
    queries: Sequence[PreparedQueryRecord],
    units: Iterable[RetrievalUnitRecord],
) -> list[RelevanceJudgementRecord]:
    units_by_document: dict[str, list[RetrievalUnitRecord]] = {}
    for unit in units:
        units_by_document.setdefault(unit.document_id, []).append(unit)

    judgements: list[RelevanceJudgementRecord] = []
    for query in queries:
        evidence = set(query.evidence_union_paragraph_ids)
        for unit in units_by_document.get(query.document_id, []):
            matched = sorted(evidence.intersection(unit.overlapping_paragraph_ids))
            if not matched:
                continue
            judgements.append(
                RelevanceJudgementRecord(
                    query_id=query.query_id,
                    document_id=query.document_id,
                    segmentation_plan=unit.segmentation_plan,
                    retrieval_unit_id=unit.retrieval_unit_id,
                    relevance=1,
                    matched_evidence_paragraph_ids=matched,
                )
            )
    return judgements
