from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from sclc.config import DocumentConfig
from sclc.data.schema import (
    AnswerAnnotation,
    CharacterSpan,
    DocumentRecord,
    EvidenceItem,
    ParagraphRecord,
    QueryRecord,
    SectionRecord,
)

_WHITESPACE = re.compile(r"\s+")
_FLOAT_PREFIX = "FLOAT SELECTED"


def normalize_text(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def _normalised_heading(text: str) -> str:
    return normalize_text(text).casefold().rstrip(":")


def _is_reference_heading(heading: str, config: DocumentConfig) -> bool:
    candidate = _normalised_heading(heading)
    return candidate in {_normalised_heading(item) for item in config.reference_headings}


def _parse_hierarchy(heading: str, separator: str) -> list[str]:
    parts = [normalize_text(part) for part in heading.split(separator)]
    return [part for part in parts if part]


def _current_position(buffer: list[str]) -> int:
    return sum(len(part) for part in buffer)


def _append_piece(buffer: list[str], piece: str) -> CharacterSpan:
    start = _current_position(buffer)
    buffer.append(piece)
    end = start + len(piece)
    return CharacterSpan(start=start, end=end)


def _extract_answer(answer: dict[str, Any], annotation_id: str | None) -> AnswerAnnotation:
    evidence_texts = answer.get("evidence") or []
    highlighted = answer.get("highlighted_evidence") or []
    return AnswerAnnotation(
        annotation_id=annotation_id,
        unanswerable=bool(answer.get("unanswerable", False)),
        extractive_spans=[str(value) for value in (answer.get("extractive_spans") or [])],
        yes_no=answer.get("yes_no"),
        free_form_answer=str(answer.get("free_form_answer") or ""),
        evidence=[
            EvidenceItem(
                text=str(text),
                is_float=normalize_text(str(text)).startswith(_FLOAT_PREFIX),
                alignment_status="pending",
            )
            for text in evidence_texts
        ],
        highlighted_evidence=[str(value) for value in highlighted],
    )


def _extract_queries(qas: dict[str, Any]) -> list[QueryRecord]:
    questions = qas.get("question") or []
    question_ids = qas.get("question_id") or []
    answer_groups = qas.get("answers") or []

    queries: list[QueryRecord] = []
    for query_index, question in enumerate(questions):
        query_id = (
            str(question_ids[query_index])
            if query_index < len(question_ids)
            else f"query_{query_index:04d}"
        )
        group = answer_groups[query_index] if query_index < len(answer_groups) else {}
        raw_annotations = group.get("answer") or []
        annotation_ids = group.get("annotation_id") or []

        answers = [
            _extract_answer(
                answer=dict(raw_answer),
                annotation_id=(
                    str(annotation_ids[index]) if index < len(annotation_ids) else None
                ),
            )
            for index, raw_answer in enumerate(raw_annotations)
        ]
        queries.append(
            QueryRecord(query_id=query_id, question=normalize_text(str(question)), answers=answers)
        )
    return queries


def _align_evidence(document: DocumentRecord) -> None:
    paragraph_lookup: dict[str, list[str]] = defaultdict(list)
    for paragraph in document.paragraphs:
        paragraph_lookup[normalize_text(paragraph.text)].append(paragraph.paragraph_id)

    for query in document.queries:
        for answer in query.answers:
            for evidence in answer.evidence:
                if evidence.is_float:
                    evidence.alignment_status = "non_text_float"
                    continue

                key = normalize_text(evidence.text)
                matches = paragraph_lookup.get(key, [])
                evidence.matched_paragraph_ids = list(matches)

                if not key:
                    evidence.alignment_status = "empty"
                elif len(matches) == 1:
                    evidence.alignment_status = "matched_unique"
                elif len(matches) > 1:
                    evidence.alignment_status = "matched_ambiguous"
                else:
                    evidence.alignment_status = "unmatched"


def reconstruct_document(
    raw: dict[str, Any],
    split: str,
    config: DocumentConfig,
) -> DocumentRecord:
    document_id = str(raw["id"])
    title = normalize_text(str(raw.get("title") or ""))
    abstract = normalize_text(str(raw.get("abstract") or ""))

    full_text = raw.get("full_text") or {}
    headings = list(full_text.get("section_name") or [])
    section_paragraphs = list(full_text.get("paragraphs") or [])

    buffer: list[str] = []
    sections: list[SectionRecord] = []
    paragraphs: list[ParagraphRecord] = []

    def add_section(
        *,
        source_section_index: int,
        heading: str,
        paragraph_texts: list[str],
        is_abstract: bool = False,
    ) -> None:
        section_id = f"{document_id}:section:{len(sections):04d}"
        hierarchy = _parse_hierarchy(heading, config.hierarchy_separator)
        top_level = hierarchy[0] if hierarchy else heading

        if buffer:
            buffer.append(config.section_separator)

        section_start = _current_position(buffer)
        paragraph_ids: list[str] = []

        if heading:
            buffer.append(heading)
            buffer.append(config.heading_separator)

        for paragraph_index, raw_paragraph in enumerate(paragraph_texts):
            paragraph_text = normalize_text(str(raw_paragraph))
            if not paragraph_text:
                continue
            if paragraph_ids:
                buffer.append(config.paragraph_separator)

            paragraph_id = f"{document_id}:paragraph:{len(paragraphs):05d}"
            paragraph_span = _append_piece(buffer, paragraph_text)
            paragraphs.append(
                ParagraphRecord(
                    paragraph_id=paragraph_id,
                    section_id=section_id,
                    source_section_index=source_section_index,
                    source_paragraph_index=paragraph_index,
                    text=paragraph_text,
                    span=paragraph_span,
                )
            )
            paragraph_ids.append(paragraph_id)

        section_end = _current_position(buffer)
        if paragraph_ids:
            sections.append(
                SectionRecord(
                    section_id=section_id,
                    source_section_index=source_section_index,
                    heading=heading,
                    top_level_heading=top_level,
                    hierarchy=hierarchy,
                    span=CharacterSpan(start=section_start, end=section_end),
                    paragraph_ids=paragraph_ids,
                    is_abstract=is_abstract,
                )
            )

    if config.include_abstract and abstract:
        abstract_paragraphs: list[str] = []
        if config.include_title and title:
            abstract_paragraphs.append(title)
        abstract_paragraphs.append(abstract)
        add_section(
            source_section_index=-1,
            heading=config.abstract_heading,
            paragraph_texts=abstract_paragraphs,
            is_abstract=True,
        )
    elif config.include_title and title:
        add_section(
            source_section_index=-1,
            heading="Title",
            paragraph_texts=[title],
            is_abstract=False,
        )

    for index, raw_heading in enumerate(headings):
        heading = normalize_text(str(raw_heading))
        hierarchy = _parse_hierarchy(heading, config.hierarchy_separator)
        top_level_heading = hierarchy[0] if hierarchy else heading

        if config.remove_reference_sections and _is_reference_heading(
            top_level_heading, config
        ):
            continue

        raw_paragraphs = section_paragraphs[index] if index < len(section_paragraphs) else []
        add_section(
            source_section_index=index,
            heading=heading,
            paragraph_texts=[str(value) for value in raw_paragraphs],
        )

    document = DocumentRecord(
        document_id=document_id,
        split=split,
        title=title,
        abstract=abstract,
        text="".join(buffer),
        sections=sections,
        paragraphs=paragraphs,
        queries=_extract_queries(dict(raw.get("qas") or {})),
    )
    _align_evidence(document)
    return document
