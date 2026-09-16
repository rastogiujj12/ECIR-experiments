from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from sclc.data.schema import (
    CharacterSpan,
    DocumentRecord,
    RetrievalUnitRecord,
    TopLevelSectionRecord,
)


class OffsetTokenizer(Protocol):
    is_fast: bool

    def __call__(self, text: str, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class TokenisedScope:
    input_ids: list[int]
    offsets: list[tuple[int, int]]


@dataclass(frozen=True)
class ChunkingStats:
    scope_count: int
    expected_tokens: int
    emitted_tokens: int
    adjusted_boundaries: int
    zero_length_offsets: int


def spans_overlap(first: CharacterSpan, second: CharacterSpan) -> bool:
    return first.start < second.end and second.start < first.end


def span_contains(container: CharacterSpan, contained: CharacterSpan) -> bool:
    return container.start <= contained.start and contained.end <= container.end


def _tokenise(tokenizer: OffsetTokenizer, text: str) -> TokenisedScope:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )
    input_ids = list(encoded["input_ids"])
    offsets = [tuple(offset) for offset in encoded["offset_mapping"]]
    if len(input_ids) != len(offsets):
        raise RuntimeError("Tokenizer returned different input-id and offset lengths")
    return TokenisedScope(input_ids=input_ids, offsets=offsets)


def _standalone_token_count(tokenizer: OffsetTokenizer, text: str) -> int:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )
    return len(encoded["input_ids"])


def build_top_level_sections(
    document: DocumentRecord,
    analysis_set: str,
) -> list[TopLevelSectionRecord]:
    """Merge contiguous subsection records under their top-level heading.

    QASPER stores hierarchy in headings such as ``Methods ::: Data``. The
    dissertation treats the top-level section as the contextual boundary, so
    contiguous records sharing the same top-level heading are merged here.
    """
    parents: list[TopLevelSectionRecord] = []

    for section in document.sections:
        normalised = section.top_level_heading.casefold().strip()
        previous = parents[-1] if parents else None
        previous_key = previous.heading.casefold().strip() if previous else None

        if previous is not None and previous_key == normalised:
            previous.span.end = section.span.end
            previous.source_section_ids.append(section.section_id)
            previous.text = document.text[previous.span.start : previous.span.end]
            continue

        parent_id = f"{document.document_id}:top_section:{len(parents):04d}"
        parents.append(
            TopLevelSectionRecord(
                parent_section_id=parent_id,
                document_id=document.document_id,
                analysis_set=analysis_set,
                heading=section.top_level_heading,
                span=CharacterSpan(start=section.span.start, end=section.span.end),
                source_section_ids=[section.section_id],
                text=document.text[section.span.start : section.span.end],
            )
        )

    return parents


def _overlapping_ids(
    span: CharacterSpan,
    records: Iterable[Any],
    id_attribute: str,
) -> list[str]:
    return [
        str(getattr(record, id_attribute))
        for record in records
        if spans_overlap(span, record.span)
    ]



def _overlapping_paragraph_ids(
    span: CharacterSpan,
    document: DocumentRecord,
) -> list[str]:
    """Return paragraphs sharing at least one non-whitespace character with a unit."""
    overlapping: list[str] = []
    for paragraph in document.paragraphs:
        start = max(span.start, paragraph.span.start)
        end = min(span.end, paragraph.span.end)
        if end <= start:
            continue
        if any(not character.isspace() for character in document.text[start:end]):
            overlapping.append(paragraph.paragraph_id)
    return overlapping

def _contained_ids(
    span: CharacterSpan,
    records: Iterable[Any],
    id_attribute: str,
) -> list[str]:
    return [
        str(getattr(record, id_attribute))
        for record in records
        if span_contains(span, record.span)
    ]


def _chunk_scope(
    *,
    document: DocumentRecord,
    analysis_set: str,
    segmentation_plan: str,
    tokenizer: OffsetTokenizer,
    chunk_size_tokens: int,
    overlap_tokens: int,
    scope_text: str,
    scope_character_start: int,
    unit_index_start: int,
    parent_section: TopLevelSectionRecord | None,
    top_level_sections: Sequence[TopLevelSectionRecord],
) -> tuple[list[RetrievalUnitRecord], ChunkingStats]:
    tokenised = _tokenise(tokenizer, scope_text)

    # Some fast tokenizers represent trailing whitespace as a real token with
    # a non-empty character offset. Such a token cannot form a meaningful
    # retrieval unit by itself and can also leave dense target pooling with no
    # non-whitespace content. Trim only trailing whitespace-only tokens;
    # whitespace occurring inside a real chunk remains part of that chunk.
    while tokenised.offsets:
        start, end = tokenised.offsets[-1]
        if end <= start or not scope_text[start:end].strip():
            tokenised = TokenisedScope(
                input_ids=tokenised.input_ids[:-1],
                offsets=tokenised.offsets[:-1],
            )
            continue
        break

    valid_positions = [
        index for index, (start, end) in enumerate(tokenised.offsets) if end > start
    ]
    zero_length_offsets = len(tokenised.offsets) - len(valid_positions)

    if not valid_positions:
        return [], ChunkingStats(
            scope_count=1,
            expected_tokens=0,
            emitted_tokens=0,
            adjusted_boundaries=0,
            zero_length_offsets=zero_length_offsets,
        )

    # The production tokenizers should not emit zero-length offsets when
    # special tokens are disabled. Failing explicitly is safer than silently
    # losing a model token from the canonical segmentation.
    if zero_length_offsets:
        raise RuntimeError(
            f"Tokenizer emitted {zero_length_offsets} zero-length offsets for "
            f"document {document.document_id}."
        )

    units: list[RetrievalUnitRecord] = []
    token_cursor = 0
    adjusted_boundaries = 0
    while token_cursor < len(tokenised.offsets):
        tentative_end = min(token_cursor + chunk_size_tokens, len(tokenised.offsets))
        token_end = tentative_end

        while token_end > token_cursor:
            local_start = tokenised.offsets[token_cursor][0]
            local_end = tokenised.offsets[token_end - 1][1]
            candidate_text = scope_text[local_start:local_end]
            if _standalone_token_count(tokenizer, candidate_text) <= chunk_size_tokens:
                break
            token_end -= 1
            adjusted_boundaries += 1

        if token_end <= token_cursor:
            raise RuntimeError(
                f"Could not create a non-empty chunk for {document.document_id} "
                f"at scope token {token_cursor}."
            )

        local_start = tokenised.offsets[token_cursor][0]
        local_end = tokenised.offsets[token_end - 1][1]
        absolute_span = CharacterSpan(
            start=scope_character_start + local_start,
            end=scope_character_start + local_end,
        )
        chunk_text = document.text[absolute_span.start : absolute_span.end]
        standalone_count = _standalone_token_count(tokenizer, chunk_text)

        unit_index = unit_index_start + len(units)
        unit_id = (
            f"{document.document_id}:{segmentation_plan}:"
            f"c{chunk_size_tokens}:{unit_index:05d}"
        )
        overlapping_sections = _overlapping_ids(
            absolute_span, document.sections, "section_id"
        )
        overlapping_parents = _overlapping_ids(
            absolute_span, top_level_sections, "parent_section_id"
        )
        overlapping_paragraphs = _overlapping_paragraph_ids(
            absolute_span, document
        )
        contained_paragraphs = _contained_ids(
            absolute_span, document.paragraphs, "paragraph_id"
        )

        units.append(
            RetrievalUnitRecord(
                retrieval_unit_id=unit_id,
                document_id=document.document_id,
                analysis_set=analysis_set,
                segmentation_plan=segmentation_plan,
                unit_index=unit_index,
                span=absolute_span,
                text=chunk_text,
                token_count=standalone_count,
                scope_token_start=token_cursor,
                scope_token_end=token_end,
                parent_section_id=(
                    parent_section.parent_section_id if parent_section else None
                ),
                parent_section_heading=(parent_section.heading if parent_section else None),
                source_section_ids=(
                    list(parent_section.source_section_ids) if parent_section else []
                ),
                overlapping_section_ids=overlapping_sections,
                overlapping_parent_section_ids=overlapping_parents,
                overlapping_paragraph_ids=overlapping_paragraphs,
                fully_contained_paragraph_ids=contained_paragraphs,
            )
        )

        if token_end == len(tokenised.offsets):
            break
        token_cursor = token_end - overlap_tokens

    emitted_tokens = sum(unit.scope_token_end - unit.scope_token_start for unit in units)
    return units, ChunkingStats(
        scope_count=1,
        expected_tokens=len(tokenised.offsets),
        emitted_tokens=emitted_tokens,
        adjusted_boundaries=adjusted_boundaries,
        zero_length_offsets=zero_length_offsets,
    )


def build_continuous_units(
    document: DocumentRecord,
    analysis_set: str,
    tokenizer: OffsetTokenizer,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> tuple[list[RetrievalUnitRecord], ChunkingStats]:
    parents = build_top_level_sections(document, analysis_set)
    return _chunk_scope(
        document=document,
        analysis_set=analysis_set,
        segmentation_plan="continuous",
        tokenizer=tokenizer,
        chunk_size_tokens=chunk_size_tokens,
        overlap_tokens=overlap_tokens,
        scope_text=document.text,
        scope_character_start=0,
        unit_index_start=0,
        parent_section=None,
        top_level_sections=parents,
    )


def build_section_bounded_units(
    document: DocumentRecord,
    analysis_set: str,
    tokenizer: OffsetTokenizer,
    chunk_size_tokens: int,
    overlap_tokens: int,
    parent_sections: Sequence[TopLevelSectionRecord] | None = None,
) -> tuple[list[RetrievalUnitRecord], ChunkingStats]:
    parents = list(parent_sections or build_top_level_sections(document, analysis_set))
    all_units: list[RetrievalUnitRecord] = []
    total_expected = 0
    total_emitted = 0
    total_adjusted = 0
    total_zero_offsets = 0

    for parent in parents:
        units, stats = _chunk_scope(
            document=document,
            analysis_set=analysis_set,
            segmentation_plan="section_bounded",
            tokenizer=tokenizer,
            chunk_size_tokens=chunk_size_tokens,
            overlap_tokens=overlap_tokens,
            scope_text=parent.text,
            scope_character_start=parent.span.start,
            unit_index_start=len(all_units),
            parent_section=parent,
            top_level_sections=parents,
        )
        all_units.extend(units)
        total_expected += stats.expected_tokens
        total_emitted += stats.emitted_tokens
        total_adjusted += stats.adjusted_boundaries
        total_zero_offsets += stats.zero_length_offsets

    return all_units, ChunkingStats(
        scope_count=len(parents),
        expected_tokens=total_expected,
        emitted_tokens=total_emitted,
        adjusted_boundaries=total_adjusted,
        zero_length_offsets=total_zero_offsets,
    )
